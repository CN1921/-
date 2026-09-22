# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

"""
游资战法量化策略（掘金版 - 纯掘金数据源 - 参数调优版）
三大战法：打板、龙头、超跌反弹
数据源：全部使用掘金 history_n

参数调优：
- 调仓间隔 10 → 5（每周调仓）
- 情绪评分阈值降低（更容易触发上升/高潮）
- 打板股性门槛 0.3 → 0.2
- 龙头要求 ≥3板 → ≥2板
- 持仓数 3 → 5
- 止损 -7% → -6%
"""


def init(context):
    context.max_hold = 5
    context.position_pct = 0.95
    context.hold_info = {}
    context.last_trade_date = None
    context.trade_day_count = 0
    context.rebalance_interval = 5
    context.stock_pool = []
    context.industry_map = {}
    context.pool_initialized = False

    schedule(schedule_func=daily_algo, date_rule='1d', time_rule='14:30:00')


def init_stock_pool(context):
    all_stocks = None
    try:
        all_stocks = get_symbol_infos(sec_type1=1010, sec_type2=101001, df=True)
    except Exception as e:
        print(f'[警告] get_symbol_infos 失败: {e}')

    if all_stocks is None or len(all_stocks) == 0:
        try:
            all_stocks = get_instrumentinfos(sec_types=[1], exchanges=['SHSE', 'SZSE'], df=True)
        except Exception:
            all_stocks = None

    if all_stocks is None or len(all_stocks) == 0:
        print('[错误] 无法获取股票列表')
        return False

    pool = []
    industry_map = {}
    for _, row in all_stocks.iterrows():
        symbol = str(row.get('symbol', ''))
        name = str(row.get('sec_name', ''))

        if '.' not in symbol:
            continue
        pure = symbol.split('.')[-1]

        if not (pure.startswith('60') or pure.startswith('000')):
            continue
        if 'ST' in name.upper():
            continue

        pool.append(symbol)
        industry = row.get('industry', '')
        if pd.notna(industry) and industry:
            industry_map[symbol] = industry

    context.stock_pool = pool
    context.industry_map = industry_map
    context.pool_initialized = True
    print(f'[数据] 主板股票池: {len(pool)} 只')
    return True


def calc_market_sentiment(context, now):
    import time as _time

    zt_list = []
    dt_list = []
    prev_zt_count = 0
    broken_count = 0
    max_board = 0

    total = len(context.stock_pool)
    start_time = _time.time()
    log_step = 200

    print(f'[扫描] 开始遍历 {total} 只股票...')

    for idx, symbol in enumerate(context.stock_pool):
        if (idx + 1) % log_step == 0 or (idx + 1) == total:
            elapsed = _time.time() - start_time
            pct = (idx + 1) / total * 100
            eta = elapsed / (idx + 1) * (total - idx - 1) if idx > 0 else 0
            print(f'[扫描] {idx+1}/{total} ({pct:.1f}%) '
                  f'| 涨停 {len(zt_list)} 跌停 {len(dt_list)} '
                  f'| 已用 {elapsed:.0f}s 剩余约 {eta:.0f}s')

        try:
            data = history_n(symbol, frequency='1d', count=25,
                            end_time=now, fields='close, amount',
                            skip_suspended=True, fill_missing='Last',
                            adjust=ADJUST_PREV, df=True)
            if data is None or len(data) < 5:
                continue

            close = data['close'].astype(float).values
            amount = data['amount'].astype(float).values

            gains = [(close[i] / close[i-1] - 1) for i in range(1, len(close))]
            if not gains:
                continue

            today_gain = gains[-1]
            is_zt_today = today_gain >= 0.098
            is_dt_today = today_gain <= -0.098
            is_zt_yesterday = len(gains) >= 2 and gains[-2] >= 0.098

            if is_zt_today:
                board = 1
                for j in range(len(gains)-2, -1, -1):
                    if gains[j] >= 0.098:
                        board += 1
                    else:
                        break
                max_board = max(max_board, board)
                zt_list.append({
                    'symbol': symbol,
                    'board': board,
                    'close': close[-1],
                    'amount': amount[-1] if len(amount) > 0 else 0,
                    'industry': context.industry_map.get(symbol, ''),
                })

            if is_dt_today:
                dt_list.append({
                    'symbol': symbol,
                    'close': close[-1],
                    'amount': amount[-1] if len(amount) > 0 else 0,
                })

            if is_zt_yesterday:
                prev_zt_count += 1
                if today_gain < 0.05:
                    broken_count += 1

        except Exception:
            continue

    total_time = _time.time() - start_time
    print(f'[扫描] 完成，共耗时 {total_time:.0f}s '
          f'| 涨停 {len(zt_list)} 跌停 {len(dt_list)}')

    broken_rate = broken_count / prev_zt_count if prev_zt_count > 0 else 0.5
    score = len(zt_list) * 0.01 + max_board * 0.3 - broken_rate * 2

    # 阈值调低，更容易触发上升/高潮
    if score > 1.0:
        phase = '高潮'
    elif score > 0.5:
        phase = '上升'
    elif score > 0.15:
        phase = '震荡'
    elif score > -0.3:
        phase = '退潮'
    else:
        phase = '冰点'

    return {
        'zt_list': zt_list,
        'dt_list': dt_list,
        'zt_count': len(zt_list),
        'max_board': max_board,
        'broken_rate': broken_rate,
        'score': score,
        'phase': phase
    }


def calc_sector_strength(zt_list):
    if not zt_list:
        return {}
    sector_count = {}
    for item in zt_list:
        ind = item.get('industry', '')
        if ind:
            sector_count[ind] = sector_count.get(ind, 0) + 1
    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def calc_position_pct(phase):
    # 整体仓位上调
    return {'高潮': 0.9, '上升': 0.7, '震荡': 0.5, '退潮': 0.2, '冰点': 0.0}.get(phase, 0.0)


def calc_personality(symbol, now):
    try:
        data = history_n(symbol=symbol, frequency='1d', count=60, end_time=now,
                         fields='close', adjust=ADJUST_PREV, df=True)
        if data is None or len(data) < 20:
            return 0
        df = data.copy()
        df['pct'] = df['close'].astype(float).pct_change()
        zt_idx = df[df['pct'] >= 0.098].index
        if len(zt_idx) < 2:
            return 0
        premiums = []
        for idx in zt_idx:
            pos = df.index.get_loc(idx)
            if pos + 1 < len(df):
                premiums.append(df['pct'].iloc[pos + 1])
        if not premiums:
            return 0
        premiums = np.array(premiums)
        f1 = np.mean(premiums > 0.05)
        f2 = np.mean(premiums > 0)
        f3 = min(len(zt_idx) / 10, 1.0)
        return f1 * 0.5 + f2 * 0.3 + f3 * 0.2
    except Exception:
        return 0


def strategy_limit_up(zt_list, sentiment, now):
    # 退潮期也可以轻仓打板
    if sentiment['phase'] == '冰点':
        return []
    candidates = []
    for item in zt_list:
        if item['board'] != 1:
            continue
        symbol = item['symbol']
        close = item['close']
        if close > 50:
            continue
        p_score = calc_personality(symbol, now)
        if p_score < 0.2:
            continue
        candidates.append({
            'symbol': symbol,
            'score': p_score,
            'strategy': '打板'
        })
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:5]


def strategy_dragon(zt_list, sentiment, sector_strength):
    if sentiment['phase'] not in ['上升', '高潮']:
        return []
    if not zt_list:
        return []
    zt_sorted = sorted(zt_list, key=lambda x: x['board'], reverse=True)
    top = zt_sorted[0]
    # 龙头门槛 3板 → 2板
    if top['board'] < 2:
        return []
    sector = top.get('industry', '')
    if sector and sector_strength.get(sector, 0) < 2:
        return []
    return [{'symbol': top['symbol'],
             'board': top['board'], 'strategy': '龙头'}]


def strategy_oversold(dt_list, sentiment):
    # 退潮期也允许超跌
    if sentiment['phase'] not in ['冰点', '退潮']:
        return []
    candidates = []
    for item in dt_list:
        candidates.append({
            'symbol': item['symbol'],
            'amount': item.get('amount', 0),
            'strategy': '超跌'
        })
    candidates.sort(key=lambda x: x['amount'])
    return candidates[:3]


def get_current_price(symbol, now):
    try:
        data = history_n(symbol=symbol, frequency='1d', count=1, end_time=now,
                         fields='close', adjust=ADJUST_PREV, df=False)
        if data and len(data) > 0:
            return float(data[0]['close'])
    except Exception:
        pass
    return None


def get_ma(symbol, now, period):
    try:
        data = history_n(symbol=symbol, frequency='1d', count=period, end_time=now,
                         fields='close', adjust=ADJUST_PREV, df=True)
        if data is not None and len(data) >= period:
            return float(data['close'].astype(float).mean())
    except Exception:
        pass
    return None


def manage_positions(context, now):
    positions = get_position()
    if not positions:
        return
    today = now.strftime('%Y-%m-%d')

    for pos in positions:
        sym = pos['symbol']
        cost = pos.get('vwap', 0)
        volume = pos.get('volume', 0)
        if cost <= 0 or volume <= 0:
            continue

        if sym not in context.hold_info:
            context.hold_info[sym] = {'buy_date': today, 'cost': cost,
                                      'high': cost, 'half_sold': False}
        info = context.hold_info[sym]
        info['cost'] = cost

        price = get_current_price(sym, now)
        if price is None:
            continue

        if price > info['high']:
            info['high'] = price

        try:
            buy_dt = pd.to_datetime(info['buy_date'])
            hold_days = (now - buy_dt).days
        except Exception:
            hold_days = 0

        pnl_pct = (price - cost) / cost
        drawdown = (info['high'] - price) / info['high'] if info['high'] > 0 else 0

        # 止损 -6%
        if pnl_pct < -0.06:
            order_target_percent(sym, percent=0, order_type=OrderType_Limit,
                                 position_side=PositionSide_Long, price=price)
            print(f'[止损] {sym} @ {price:.2f}，亏损 {pnl_pct:.2%}')
            context.hold_info.pop(sym, None)
            continue

        # 趋势破坏：跌破10日线且持有≥3天（从5天降到3天）
        ma10 = get_ma(sym, now, 10)
        if ma10 and price < ma10 and hold_days >= 3:
            order_target_percent(sym, percent=0, order_type=OrderType_Limit,
                                 position_side=PositionSide_Long, price=price)
            print(f'[趋势] {sym} @ {price:.2f}，跌破10日线，盈亏 {pnl_pct:.2%}')
            context.hold_info.pop(sym, None)
            continue

        # 移动止盈：回撤8%且盈利超5%
        if drawdown > 0.08 and pnl_pct > 0.05:
            order_target_percent(sym, percent=0, order_type=OrderType_Limit,
                                 position_side=PositionSide_Long, price=price)
            print(f'[止盈] {sym} @ {price:.2f}，回撤 {drawdown:.2%}，盈利 {pnl_pct:.2%}')
            context.hold_info.pop(sym, None)
            continue

        # 减半止盈：浮盈超15%（从20%降到15%）
        if pnl_pct > 0.15 and not info['half_sold']:
            half_vol = int(volume / 2 / 100) * 100
            if half_vol > 0:
                order_volume(sym, volume=half_vol, side=OrderSide_Sell,
                             order_type=OrderType_Limit,
                             position_effect=PositionEffect_Close, price=price)
                info['half_sold'] = True
                print(f'[减半] {sym} @ {price:.2f}，盈利 {pnl_pct:.2%}')
            continue

        print(f'[持有] {sym} @ {price:.2f}，盈亏 {pnl_pct:.2%}，{hold_days}天')


def daily_algo(context):
    now = context.now
    today = now.strftime('%Y-%m-%d')

    if not context.pool_initialized:
        if not init_stock_pool(context):
            return

    # 大盘风控：指数跌破20日线清仓
    index_data = history_n(symbol='SHSE.000300', frequency='1d', count=20,
                          end_time=now, fields='close', df=True)
    if index_data is not None and len(index_data) >= 20:
        ma20 = index_data['close'].astype(float).mean()
        current = float(index_data['close'].iloc[-1])
        if current < ma20:
            positions = get_position()
            if positions:
                print('[风控] 大盘走弱，清仓')
                for pos in positions:
                    p = get_current_price(pos['symbol'], now)
                    if p:
                        order_target_percent(pos['symbol'], percent=0,
                                             order_type=OrderType_Limit,
                                             position_side=PositionSide_Long, price=p)
                context.hold_info.clear()
            return

    manage_positions(context, now)

    if context.last_trade_date == today:
        return
    context.trade_day_count += 1
    if context.trade_day_count % context.rebalance_interval != 0:
        return

    sentiment = calc_market_sentiment(context, now)
    print(f'[情绪] {sentiment["phase"]} | 评分 {sentiment["score"]:.2f} | 涨停 {sentiment["zt_count"]}家 | 最高 {sentiment["max_board"]}板 | 炸板率 {sentiment["broken_rate"]:.2f}')

    zt_list = sentiment['zt_list']
    dt_list = sentiment['dt_list']
    sector_strength = calc_sector_strength(zt_list)
    if sector_strength:
        print(f'[板块] {list(sector_strength.items())[:5]}')

    signals = []
    signals += strategy_limit_up(zt_list, sentiment, now)
    signals += strategy_dragon(zt_list, sentiment, sector_strength)
    if sentiment['phase'] in ['冰点', '退潮']:
        signals += strategy_oversold(dt_list, sentiment)

    if not signals:
        print('[信号] 无交易信号')
        context.last_trade_date = today
        return

    seen = set()
    unique_signals = []
    for s in signals:
        if s['symbol'] not in seen:
            seen.add(s['symbol'])
            unique_signals.append(s)

    print(f'[信号] 共 {len(unique_signals)} 只')
    for s in unique_signals:
        print(f'  {s["symbol"]} ({s["strategy"]})')

    positions = get_position()
    target_symbols = [s['symbol'] for s in unique_signals[:context.max_hold]]
    for pos in positions:
        if pos['symbol'] not in target_symbols:
            p = get_current_price(pos['symbol'], now)
            if p:
                order_target_percent(pos['symbol'], percent=0, order_type=OrderType_Limit,
                                     position_side=PositionSide_Long, price=p)
                print(f'[平仓] {pos["symbol"]} @ {p:.2f}')
                context.hold_info.pop(pos['symbol'], None)

    positions = get_position()
    current_symbols = [p['symbol'] for p in positions]
    current_hold = len(positions)
    position_pct = calc_position_pct(sentiment['phase'])

    for s in unique_signals:
        if current_hold >= context.max_hold:
            break
        if s['symbol'] in current_symbols:
            continue

        p = get_current_price(s['symbol'], now)
        if p is None or p <= 0:
            continue

        target_pct = position_pct / context.max_hold
        acc = context.account()
        cash_raw = acc.cash
        cash_val = float(cash_raw.get('available', 0)) if isinstance(cash_raw, dict) else float(cash_raw)
        target_amount = cash_val * target_pct
        target_vol = int(target_amount / p / 100) * 100
        if target_vol <= 0:
            continue

        order_target_volume(s['symbol'], volume=target_vol,
                            order_type=OrderType_Limit,
                            position_side=PositionSide_Long, price=p)
        print(f'[买入] {s["symbol"]} @ {p:.2f} 仓位 {target_pct:.2%} 数量 {target_vol}股 ({s["strategy"]})')

        context.hold_info[s['symbol']] = {
            'buy_date': today, 'cost': p, 'high': p, 'half_sold': False
        }
        current_hold += 1

    context.last_trade_date = today


def on_order_status(context, order):
    if order['status'] == 3:
        side = '买入' if order['side'] == 1 else '卖出'
        print(f'[成交] {order["symbol"]} {side} 价格 {order["price"]:.2f} 数量 {order["volume"]}')


def on_backtest_finished(context, indicator):
    print('=' * 60)
    print('[完成] 回测结束')
    try:
        print(f'  累计收益: {indicator["pnl_ratio"]:.2%}')
        print(f'  最大回撤: {indicator["max_drawdown"]:.2%}')
        print(f'  夏普比率: {indicator["sharpe_ratio"]:.2f}')
    except Exception:
        pass
    print('=' * 60)





if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
        '''
    run(strategy_id='1022b98e-b5a8-11f1-9e3a-80fa5b6f62b4',
        filename='main.py',
        mode=MODE_LIVE,
        token='1a1573a353ce97ddbb3f6abe75c4d45619119bc9',
        backtest_start_time='2026-09-22 16:00:00',
        backtest_end_time='2026-09-18 09:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.000025,
        backtest_slippage_ratio=0.0003,
        backtest_match_mode=1)
