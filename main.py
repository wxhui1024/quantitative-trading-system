"""
A股量化选股脚本 - 涨停双响炮/涨停启动->爆量->缩量洗盘形态策略
重写版本 - 修正时间窗口错位和条件过严问题
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from tqdm import tqdm
import time
import os
from typing import Dict, List, Tuple
import warnings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
warnings.filterwarnings('ignore')

class Strategy:
    """
    涨停双响炮/涨停启动->爆量->缩量洗盘形态策略类
    """
    
    def __init__(self):
        """初始化策略参数"""
        self.min_market_cap = 50  # 最小市值 50亿
        self.max_market_cap = 500  # 最大市值 500亿
        self.volume_threshold = 1.5  # 涨停次日放量阈值
        self.ma5_volume_threshold = 0.6  # T日MA5量相对于爆量日的比例阈值
        self.lookback_period = 25  # 回溯期
        self.start_event_window_start = 15  # 启动事件窗口起始 (T-15)
        self.start_event_window_end = 3  # 启动事件窗口结束 (T-3)
        self.session = self.create_session()
        
    def create_session(self):
        """创建带重试机制的会话"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_trading_dates(self, end_date: str, n_days: int) -> list:
        """获取n个交易日的日期列表"""
        try:
            trading_dates = ak.tool_trade_date_hist_sina()
            trading_dates = trading_dates[trading_dates['trade_date'] <= end_date]
            return trading_dates['trade_date'].tail(n_days).tolist()
        except Exception as e:
            print(f"获取交易日历失败: {e}")
            # 如果获取失败，返回一个近似的日期列表
            dates = []
            current_date = datetime.strptime(end_date, '%Y-%m-%d')
            for i in range(n_days):
                dates.append((current_date - timedelta(days=i)).strftime('%Y-%m-%d'))
            return dates

    def get_stock_pool(self) -> pd.DataFrame:
        """
        获取基础股票池 (沪深主板 + 创业板，排除ST、科创板、北交所)
        """
        print("正在获取股票列表...")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # 获取A股实时行情数据
                stock_zh_a_spot = ak.stock_zh_a_spot_em()
                
                # 筛选条件：
                # 1. 排除ST股票
                # 2. 排除科创板(688)、北交所(4/8开头)
                # 3. 只保留主板和创业板
                # 4. 市值筛选
                stock_pool = stock_zh_a_spot[
                    (~stock_zh_a_spot['代码'].str.startswith(('688', '4', '8'))) &
                    (~stock_zh_a_spot['名称'].str.contains('ST')) &
                    (stock_zh_a_spot['总市值'] >= self.min_market_cap * 1e8) &
                    (stock_zh_a_spot['总市值'] <= self.max_market_cap * 1e8)
                ].copy()
                
                print(f"基础股票池共 {len(stock_pool)} 只股票")
                return stock_pool
            except Exception as e:
                print(f"获取股票池失败，第 {attempt + 1} 次尝试: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 递增等待时间
                else:
                    print("获取股票池失败，返回空数据")
                    return pd.DataFrame()
        return pd.DataFrame()
    
    def check_pattern_single(self, symbol: str) -> Dict:
        """
        检查单只股票是否满足涨停双响炮/涨停启动->爆量->缩量洗盘形态
        """
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 获取股票历史数据 - 获取更多数据以确保有25个交易日
                    stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                    
                    if len(stock_zh_a_hist) < self.lookback_period:
                        return None
                    
                    # 只取最近25天的数据
                    hist_data = stock_zh_a_hist.tail(self.lookback_period).reset_index(drop=True)
                    
                    # 计算涨跌幅
                    hist_data['pct_change'] = hist_data['收盘'].pct_change() * 100
                    
                    # 识别涨停日 (涨幅 >= 9.8% 为涨停，考虑误差)
                    limit_up_mask = hist_data['pct_change'] >= 9.8
                    
                    # 找到考察区间[T-15, T-3]内的涨停事件
                    start_event_start_idx = max(0, len(hist_data) - self.start_event_window_start - 1)
                    start_event_end_idx = len(hist_data) - self.start_event_window_end
                    
                    # 在[T-15, T-3]区间寻找启动事件
                    limit_up_indices = []
                    for i in range(start_event_start_idx, start_event_end_idx):
                        if limit_up_mask.iloc[i]:
                            limit_up_indices.append(i)
                    
                    # 检查是否有多余的涨停（在考察区间外的涨停是允许的，但考察区间内只能有一次启动事件）
                    if len(limit_up_indices) == 0:
                        return None  # 没有找到启动事件
                    
                    # 检查启动事件是否为单日涨停或连续两日涨停
                    # 从找到的涨停日中判断是否存在连续涨停或单日涨停
                    start_event_indices = []
                    i = 0
                    while i < len(limit_up_indices):
                        current_start = limit_up_indices[i]
                        
                        # 检查是否是连续涨停
                        if i + 1 < len(limit_up_indices) and limit_up_indices[i+1] == current_start + 1:
                            # 连续两日涨停
                            start_event_indices = [current_start, current_start + 1]
                            i += 2  # 跳过下一个
                        else:
                            # 单日涨停
                            start_event_indices = [current_start]
                            i += 1
                        
                        # 验证这个启动事件是否符合严格排他性
                        # 检查在考察区间内除了这个启动事件外是否有其他涨停
                        other_limit_ups = [idx for idx in limit_up_indices if idx not in start_event_indices]
                        if len(other_limit_ups) == 0:
                            # 满足严格排他性
                            final_start_idx = max(start_event_indices)
                            break
                        else:
                            # 不满足严格排他性，继续寻找下一个可能的启动事件
                            start_event_indices = []
                    
                    if not start_event_indices:
                        return None  # 没有找到符合条件的启动事件
                    
                    # 确定最终的启动日（最后一个涨停日）
                    final_start_idx = max(start_event_indices)
                    
                    # 检查启动事件后一天（爆量日）
                    burst_day_idx = final_start_idx + 1
                    if burst_day_idx >= len(hist_data):
                        return None
                    
                    # 检查爆量条件
                    start_vol = hist_data['成交量'].iloc[final_start_idx]
                    burst_vol = hist_data['成交量'].iloc[burst_day_idx]
                    
                    volume_ratio = burst_vol / start_vol if start_vol > 0 else 0
                    if volume_ratio < self.volume_threshold:
                        return None
                    
                    # 检查从爆量日之后直到T日的缩量洗盘条件
                    # 计算5日成交量均线
                    hist_data['MA5_Volume'] = hist_data['成交量'].rolling(window=5).mean()
                    
                    # 检查从爆量日+1开始到T日的每一天是否满足缩量洗盘条件
                    wash_start_idx = burst_day_idx + 1
                    has_higher_volume_after_burst = False
                    
                    for k in range(wash_start_idx, len(hist_data)):
                        if hist_data['成交量'].iloc[k] > burst_vol:
                            has_higher_volume_after_burst = True
                            break
                    
                    if has_higher_volume_after_burst:
                        return None
                    
                    # 检查T日的MA5量是否满足条件
                    ma5_today = hist_data['MA5_Volume'].iloc[-1]
                    if ma5_today >= self.ma5_volume_threshold * burst_vol:
                        return None
                    
                    # 计算洗盘天数
                    wash_days = len(hist_data) - 1 - burst_day_idx
                    
                    # 计算当前量能水位
                    current_volume_level = ma5_today / burst_vol
                    
                    # 返回符合条件的股票信息
                    current_price = hist_data['收盘'].iloc[-1]
                    market_cap = hist_data['总市值'].iloc[-1] / 1e8  # 转换为亿
                    limit_up_date = hist_data['日期'].iloc[final_start_idx]
                    
                    return {
                        'code': symbol,
                        'name': '',  # 后续补充
                        'current_price': round(current_price, 2),
                        'market_cap': round(market_cap, 2),
                        'limit_up_date': limit_up_date,
                        'volume_ratio': round(volume_ratio, 2),
                        'industry': '',  # 后续补充
                        'buy_point': round(current_price * 0.98, 2),  # 建议买入点（当前价下方2%）
                        'stop_loss': round(current_price * 0.90, 2),  # 止损位（当前价下方10%）
                        'take_profit': round(current_price * 1.20, 2),  # 止盈位（当前价上方20%）
                        'wash_days': wash_days,  # 洗盘天数
                        'volume_level': round(current_volume_level * 100, 2)  # 当前量能水位（百分比）
                    }
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"处理股票 {symbol} 时出错: {e}")
                        return None
                    time.sleep(1 * (attempt + 1))  # 递增等待时间
                    
        except Exception as e:
            print(f"处理股票 {symbol} 时出错: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Tuple[str, str]:
        """
        获取股票名称和行业信息
        """
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 获取A股实时行情数据
                    stock_zh_a_spot = ak.stock_zh_a_spot_em()
                    stock_info = stock_zh_a_spot[stock_zh_a_spot['代码'] == symbol]
                    
                    if len(stock_info) > 0:
                        name = stock_info['名称'].iloc[0]
                        industry = stock_info['行业'].iloc[0] if '行业' in stock_info.columns else ''
                        return name, industry
                    else:
                        return '', ''
                except Exception as e:
                    if attempt == max_retries - 1:
                        return '', ''
                    time.sleep(1 * (attempt + 1))
        except:
            return '', ''
    
    def run(self) -> pd.DataFrame:
        """
        执行选股策略（使用多线程）
        """
        print("开始执行选股策略...")
        
        # 获取股票池
        stock_pool = self.get_stock_pool()
        
        if stock_pool.empty:
            print("未能获取到股票池，策略执行终止")
            return pd.DataFrame()
        
        # 使用多线程处理
        results = []
        max_workers = 8  # 调整为8个线程，避免过于频繁的请求
        
        print(f"开始多线程扫描 {len(stock_pool)} 只股票...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_symbol = {
                executor.submit(self.check_pattern_single, row['代码']): row['代码'] 
                for _, row in stock_pool.iterrows()
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_symbol), total=len(future_to_symbol), desc="扫描股票"):
                symbol = future_to_symbol[future]
                try:
                    pattern_result = future.result()
                    if pattern_result:
                        # 补充股票名称和行业
                        name, industry = self.get_stock_info(symbol)
                        pattern_result['name'] = name
                        pattern_result['industry'] = industry
                        results.append(pattern_result)
                except Exception as e:
                    print(f"处理股票 {symbol} 时出错: {e}")
                
                # 防止请求过快
                time.sleep(0.2)  # 增加延时，减少请求频率
        
        if results:
            df_results = pd.DataFrame(results)
            return df_results
        else:
            return pd.DataFrame()


def generate_obsidian_note(df: pd.DataFrame):
    """
    生成Obsidian格式的笔记
    """
    if df.empty:
        print("没有找到符合条件的股票")
        # 创建空的报告文件
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"stock_selection_{today}.md"
        
        # YAML frontmatter
        frontmatter = {
            'date': today,
            'tags': ['#量化选股', '#待观察'],
            'title': f'股票筛选结果 {today}'
        }
        
        md_content = "---\n"
        md_content += yaml.dump(frontmatter, allow_unicode=True)
        md_content += "---\n\n"
        md_content += f"# {today} 量化选股结果\n\n"
        md_content += "今日未找到符合策略条件的股票。\n\n"
        
        # 保存文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"空结果报告已保存至 {filename}")
        return filename
    
    # 创建Obsidian格式的笔记
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"stock_selection_{today}.md"
    
    # YAML frontmatter
    frontmatter = {
        'date': today,
        'tags': ['#量化选股', '#待观察'],
        'title': f'股票筛选结果 {today}'
    }
    
    md_content = "---\n"
    md_content += yaml.dump(frontmatter, allow_unicode=True)
    md_content += "---\n\n"
    md_content += f"# {today} 量化选股结果\n\n"
    md_content += "| 股票代码 | 股票名称 | 现价 | 总市值(亿) | 涨停日期 | 涨停次日量比 | 买入点 | 止损位 | 止盈位 | 洗盘天数 | 当前量能水位(%) | 行业板块 |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df.iterrows():
        md_content += f"| {row['code']} | {row['name']} | {row['current_price']} | {row['market_cap']} | {row['limit_up_date']} | {row['volume_ratio']} | {row['buy_point']} | {row['stop_loss']} | {row['take_profit']} | {row['wash_days']} | {row['volume_level']} | {row['industry']} |\n"
    
    # 保存文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"结果已保存至 {filename}")
    return filename


def main():
    """
    主函数
    """
    strategy = Strategy()
    results = strategy.run()
    
    if not results.empty:
        print(f"\n找到 {len(results)} 只符合条件的股票:")
        print(results[['code', 'name', 'current_price', 'market_cap', 'wash_days', 'volume_level']].to_string(index=False))
        
        # 生成Obsidian笔记
        file_path = generate_obsidian_note(results)
        
        print(f"\n选股结果已保存到本地文件: {file_path}")
        print("请注意：文件已保存到本地，您可以手动复制到Obsidian仓库中")
    else:
        print("\n未找到符合条件的股票")
        
        # 生成空结果报告
        file_path = generate_obsidian_note(results)
        
        print(f"\n选股结果已保存到本地文件: {file_path}")
        print("请注意：文件已保存到本地，您可以手动复制到Obsidian仓库中")


if __name__ == "__main__":
    main()