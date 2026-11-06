import os, sys, time, requests, pandas as pd, numpy as np

WEIGHT = {"H4":1.5,"H1":1.2,"M15":0.8,"M5":0.5}
CONTRACT_SIZE = 100
BUDGET = float(os.getenv("BUDGET","2.0")); SPREAD = float(os.getenv("SPREAD","0.3"))



def fetch_yahoo(tf):
    # Yahoo Finance 免费 K 线接口
    interval = {"M5":"5m", "M15":"15m", "H1":"1h", "H4":"1h"}[tf] # Yahoo 最小粒度5m，4h先用1h代替
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range=5d"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    ts = res.get('timestamp', [])
    quotes = res['indicators']['quote'][0]
    out = []
    for i in range(len(ts)):
        if quotes['open'][i] is not None and quotes['close'][i] is not None:
            out.append({
                "time": pd.to_datetime(ts[i], unit='s', utc=True),
                "open": quotes['open'][i], "high": quotes['high'][i],
                "low": quotes['low'][i], "close": quotes['close'][i]
            })
    df = pd.DataFrame(out)
    if len(df) < 30: raise ValueError(f"{tf} 数据不足")

    # 如果是请求 H4，就用 1h 的合成一下
    if tf == "H4":
        df['time'] = df['time'].dt.floor('4h')
        df = df.groupby('time').agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last')).reset_index()
    return df

def load_all():
    m5 = fetch_yahoo("M5"); fr = {"M5":m5}
    for tf in ("M15","H1","H4"):
        try:
            c = fetch_yahoo(tf)
            fr[tf] = c if len(c)>=30 else None
        except: fr[tf] = None
    return fr

def ind(df):
    x = df.copy()
    x["EMA20"] = x.close.ewm(span=20,adjust=False).mean()
    x["EMA50"] = x.close.ewm(span=50,adjust=False).mean()
    pc = x.close.shift(1)
    tr = pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta = x.close.diff()
    g = delta.where(delta>0,0).ewm(alpha=1/14,adjust=False).mean()
    l = (-delta.where(delta<0,0)).ewm(alpha=1/14,adjust=False).mean()
    x["RSI"] = (100-100/(1+g/l.replace(0,np.nan))).fillna(50)
    up,dn = x.high.diff(), -x.low.diff()
    pdm = np.where((up>dn)&(up>0),up,0.0); mdm = np.where((dn>up)&(dn>0),dn,0.0)
    a14 = x["ATR"].replace(0,np.nan)
    pdi = 100*pd.Series(pdm,index=x.index).ewm(alpha=1/14,adjust=False).mean()/a14
    mdi = 100*pd.Series(mdm,index=x.index).ewm(alpha=1/14,adjust=False).mean()/a14
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["ADX"] = dx.ewm(alpha=1/14,adjust=False).mean().fillna(0)
    return x

def tb(df):
    if df is None or len(df)<30: return None
    last = ind(df).iloc[-1]
    c,e20,e50 = float(last.close),float(last.EMA20),float(last.EMA50)
    if c>e20>e50: return 1.0
    if c<e20<e50: return -1.0
    return 0.5 if e20>e50 else -0.5 if e20<e50 else 0.0

def direction(fr):
    b = {tf:tb(fr.get(tf)) for tf in WEIGHT}
    av = {k:v for k,v in b.items() if v is not None}
    tw = sum(WEIGHT[k] for k in av); s = sum(WEIGHT[k]*v for k,v in av.items())
    if s==0: return 1,0,"弱",b
    ratio = min(1,abs(s)/tw) if tw else 0
    side = 1 if s>0 else -1
    strength = "强" if ratio>=0.75 else "中" if ratio>=0.45 else "弱"
    return side,ratio,strength,b

def fe(side,price,e20,atr):
    if side==1: return e20 if e20<=price-0.2*atr else price-0.2*atr
    return e20 if e20>=price+0.2*atr else price+0.2*atr

def plan(side,df,price,spread,budget):
    x = ind(df); last = x.iloc[-1]
    atr = float(last.ATR) if float(last.ATR)>0 else max(price*0.001,0.01)
    e20,rsi,adx = float(last.EMA20),float(last.RSI),float(last.ADX)
    e1 = fe(side,price,e20,atr); e2 = e1-side*0.6*atr
    sw = float(x.low.tail(16).min()) if side==1 else float(x.high.tail(16).max())
    ba = e2-side*1.5*atr; bs = sw-side*0.15*atr
    sl = min(ba,bs) if side==1 else max(ba,bs)
    sl = max(sl,e2-3*atr) if side==1 else min(sl,e2+3*atr)
    avg = (e1+e2)/2; r = abs(avg-sl); tps = [avg+side*r*m for m in (1.0,1.8,2.6)]
    r1 = abs(e1-sl)+spread
    lots = max(0.01, round(budget/(r1*CONTRACT_SIZE),2)) if r1>0 else 0.01
    return dict(e1=e1,e2=e2,sl=sl,tps=tps,rsi=rsi,adx=adx,risk1=r1,lots=lots)

def filt(side,p,strength):
    ns,f = strength,1.0
    if side==1 and p["rsi"]>75: ns = "中" if ns=="强" else "弱"; f*=0.5
    elif side==-1 and p["rsi"]<25: ns = "中" if ns=="强" else "弱"; f*=0.5
    if p["adx"]<20: f*=0.5
    return ns,f

def tg(text):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat: return False
    for _ in range(2):
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              data={"chat_id":chat,"text":text,"disable_web_page_preview":True}, timeout=12)
            if r.json().get("ok"): return True
        except: pass
    return False

if __name__ == "__main__":
    try: fr = load_all()
    except Exception as e: print(f"fetch failed: {e}"); sys.exit(0)
    m5 = fr["M5"]; price = float(m5.close.iloc[-1])
    side,ratio,strength,_ = direction(fr)
    p = plan(side, fr["M15"] if fr.get("M15") is not None else m5, price, SPREAD, BUDGET)
    strength,f = filt(side,p,strength)
    lots = max(0.01, round(p["lots"]*f,2))
    icon = "🟢" if side==1 else "🔴"; name = "BUY" if side==1 else "SELL"
    txt = (f"🥇 Kenny AI GOLD v14 (免费源)\n{icon} {name} LIMIT | 强度:{strength}\n"
           f"现价 {price:.2f} | RSI {p['rsi']:.1f} | ADX {p['adx']:.1f}\n"
           f"E1 {p['e1']:.2f} | E2 {p['e2']:.2f} | SL {p['sl']:.2f}\n"
           f"TP1 {p['tps'][0]:.2f} | TP2 {p['tps'][1]:.2f} | TP3 {p['tps'][2]:.2f}\n"
           f"💡 {lots}手 (预算${BUDGET})\n⚠️ 仅供参考")
    print("sent" if tg(txt) else "send failed")
