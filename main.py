"""
A股量化选股脚本 - 涨停启动后缩量回调策略
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
warnings.filterwarnings('ignore')

class Strategy:
    """
    涨停启动后缩量回调策略类
    """
    
    def __init__(self):
        """初始化策略参数"""
        self.min_market_cap = 50  # 最小市值 50亿
        self.max_market_cap = 500  # 最大市值 500亿
        self.volume_threshold = 1.5  # 涨停次日放量阈值
        self.max_limit_up_days = 2  # 最大涨停天数
        self.lookback_period = 10  # 回溯期
        
    def get_stock_pool(self) -> pd.DataFrame:
        """
        获取基础股票池 (沪深主板 + 创业板，排除ST、科创板、北交所)
        """
        print("正在获取股票列表...")
        
        # 获取A股实时行情数据
        stock_zh_a_spot = ak.stock_zh_a_spot_em()
        
        # 筛选条件：
        # 1. 排除ST股票
        # 2. 排除科创板(688)、北交所(4/8开头)
        # 3. 只保留主板和创业板
        stock_pool = stock_zh_a_spot[
            (~stock_zh_a_spot['代码'].str.startswith(('688', '4', '8'))) &
            (~stock_zh_a_spot['名称'].str.contains('ST')) &
            (stock_zh_a_spot['总市值'] >= self.min_market_cap * 1e8) &
            (stock_zh_a_spot['总市值'] <= self.max_market_cap * 1e8)
        ].copy()
        
        print(f"基础股票池共 {len(stock_pool)} 只股票")
        return stock_pool
    
    def check_pattern(self, symbol: str, days: int = 10) -> Dict:
        """
        检查股票是否满足涨停启动后缩量回调形态
        """
        try:
            # 获取股票历史数据
            stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
            
            if len(stock_zh_a_hist) < days:
                return None
                
            # 只取最近days天的数据
            hist_data = stock_zh_a_hist.tail(days).reset_index(drop=True)
            
            # 计算涨跌幅
            hist_data['pct_change'] = hist_data['收盘'].pct_change() * 100
            
            # 识别涨停日 (涨幅 >= 9.8% 为涨停，考虑误差)
            limit_up_mask = hist_data['pct_change'] >= 9.8
            
            # 检查涨停条件：T-3至T-10内出现1次单日涨停或连续2日涨停，且总数不超过2次
            valid_limit_up_dates = []
            consecutive_count = 0
            total_limit_up = 0
            
            for i in range(3, len(limit_up_mask)-1):  # 从T-3开始检查
                if limit_up_mask.iloc[i]:
                    total_limit_up += 1
                    if consecutive_count == 0:
                        consecutive_count = 1
                    else:
                        consecutive_count += 1
                else:
                    if consecutive_count > 0:
                        # 记录连续涨停结束
                        if consecutive_count <= 2:  # 连续涨停不超过2天
                            valid_limit_up_dates.extend(list(range(i-consecutive_count, i)))
                        consecutive_count = 0
            
            # 处理循环结束后仍处于连续状态的情况
            if consecutive_count > 0:
                if consecutive_count <= 2:
                    valid_limit_up_dates.extend(list(range(len(limit_up_mask)-consecutive_count, len(limit_up_mask))))
            
            # 检查条件
            if total_limit_up == 0 or total_limit_up > self.max_limit_up_days:
                return None
            
            # 检查是否存在连续涨停超过2天的情况
            if len(valid_limit_up_dates) != total_limit_up:
                return None
                
            # 确定最后一个涨停日
            last_limit_up_idx = max([i for i in range(len(limit_up_mask)) if limit_up_mask.iloc[i]])
            
            # 检查涨停后是否有跌停
            has_limit_down = False
            for j in range(last_limit_up_idx + 1, len(hist_data)):
                if hist_data['pct_change'].iloc[j] <= -9.8:
                    has_limit_down = True
                    break
                    
            if has_limit_down:
                return None
                
            # 检查涨停次日成交量是否放大 >= 1.5倍
            volume_burst_day_idx = last_limit_up_idx + 1
            if volume_burst_day_idx >= len(hist_data):
                return None
                
            limit_up_volume = hist_data['成交量'].iloc[last_limit_up_idx]
            burst_day_volume = hist_data['成交量'].iloc[volume_burst_day_idx]
            
            volume_ratio = burst_day_volume / limit_up_volume if limit_up_volume > 0 else 0
            
            if volume_ratio < self.volume_threshold:
                return None
                
            # 检查从爆量日后一天开始是否持续缩量
            has_higher_volume_after_burst = False
            for k in range(volume_burst_day_idx + 1, len(hist_data)):
                if hist_data['成交量'].iloc[k] >= burst_day_volume:
                    has_higher_volume_after_burst = True
                    break
                    
            if has_higher_volume_after_burst:
                return None
                
            # 返回符合条件的股票信息
            current_price = hist_data['收盘'].iloc[-1]
            market_cap = hist_data['总市值'].iloc[-1] / 1e8  # 转换为亿
            limit_up_date = hist_data['日期'].iloc[last_limit_up_idx]
            
            return {
                'code': symbol,
                'name': '',  # 后续补充
                'current_price': round(current_price, 2),
                'market_cap': round(market_cap, 2),
                'limit_up_date': limit_up_date,
                'volume_ratio': round(volume_ratio, 2),
                'industry': ''  # 后续补充
            }
            
        except Exception as e:
            print(f"处理股票 {symbol} 时出错: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Tuple[str, str]:
        """
        获取股票名称和行业信息
        """
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
        except:
            return '', ''
    
    def run(self) -> pd.DataFrame:
        """
        执行选股策略
        """
        print("开始执行选股策略...")
        
        # 获取股票池
        stock_pool = self.get_stock_pool()
        
        results = []
        
        # 遍历股票池
        for _, row in tqdm(stock_pool.iterrows(), total=len(stock_pool), desc="扫描股票"):
            symbol = row['代码']
            
            # 检查形态
            pattern_result = self.check_pattern(symbol)
            
            if pattern_result:
                # 补充股票名称和行业
                name, industry = self.get_stock_info(symbol)
                pattern_result['name'] = name
                pattern_result['industry'] = industry
                
                results.append(pattern_result)
                
            # 防止请求过快
            time.sleep(0.5)
        
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
        return
    
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
    md_content += "| 股票代码 | 股票名称 | 现价 | 总市值(亿) | 涨停日期 | 涨停次日量比 | 行业板块 |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df.iterrows():
        md_content += f"| {row['code']} | {row['name']} | {row['current_price']} | {row['market_cap']} | {row['limit_up_date']} | {row['volume_ratio']} | {row['industry']} |\n"
    
    # 保存文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"结果已保存至 {filename}")
    return filename


def auto_commit_and_push(file_path: str):
    """
    自动提交并推送结果到GitHub
    """
    try:
        import git
        from git import Repo
        import subprocess
        
        # 检查是否在git仓库中
        try:
            repo = Repo(search_parent_directories=True)
            repo_path = repo.working_tree_dir
        except:
            print("当前目录不在git仓库中，正在初始化...")
            repo = Repo.init()
            repo_path = repo.working_dir
        
        # 添加文件
        repo.index.add([file_path])
        
        # 提交
        commit_msg = f"feat: 添加 {datetime.now().strftime('%Y-%m-%d')} 量化选股结果"
        repo.index.commit(commit_msg)
        
        # 推送到远程仓库
        origin = repo.remote(name='origin')
        origin.push()
        
        print(f"文件 {file_path} 已自动提交并推送至GitHub")
        
    except ImportError:
        print("未安装GitPython库，尝试使用subprocess...")
        try:
            subprocess.run(['git', 'add', file_path], check=True)
            subprocess.run(['git', 'commit', '-m', f"feat: 添加 {datetime.now().strftime('%Y-%m-%d')} 量化选股结果"], check=True)
            subprocess.run(['git', 'push'], check=True)
            print(f"文件 {file_path} 已自动提交并推送至GitHub")
        except subprocess.CalledProcessError as e:
            print(f"自动提交失败: {e}")
    except Exception as e:
        print(f"自动提交过程出错: {e}")


def main():
    """
    主函数
    """
    strategy = Strategy()
    results = strategy.run()
    
    if not results.empty:
        print(f"\n找到 {len(results)} 只符合条件的股票:")
        print(results[['code', 'name', 'current_price', 'market_cap']].to_string(index=False))
        
        # 生成Obsidian笔记
        file_path = generate_obsidian_note(results)
        
        # 自动提交到GitHub
        if file_path:
            auto_commit_and_push(file_path)
    else:
        print("\n未找到符合条件的股票")


if __name__ == "__main__":
    main()