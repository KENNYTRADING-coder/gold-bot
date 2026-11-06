import os, sys, time, requests, pandas as pd, numpy as np

API_URL = "https://api.realmarketapi.com/api/v1/candle"
DELTA = {"M5": pd.Timedelta(minutes=5), "M15": pd.Timedelta(minutes=15),
         "H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4)}
FREQ = {"M15":"15min","H1":"1h","H4":"4h"}
WEIGHT = {"H4":1.5,"H1":1.2,"M15":0.8,"M5":0.5}
CONTRACT_SIZE = 100
BUDGET = float(os.getenv("BUDGET","2.0")); SPREAD = float(os.getenv("SPREAD","0.3"))

def env(n):
    v = os.getenv(n,"")
    if not v: sys.exit(f"missing {n}")
    return v

def gf(x,*ks):
    for k in ks:
        if k in x and x[k] is not None: return x[k]
    return None

def fetch(tf):
    r = requests.get(API_URL, params={"apiKey":env("REALMARKETAPI_KEY"),"symbolCode":"XAUUSD","timeFrame":tf},
                     headers={"User-Agent":"KennyAIGold/14-push","Accept":"application/json"}, timeout=20)
    r.raise_for_status(); p = r.json()
    rows = p.get("data", p.get("Data", [])) if isinstance(p, dict) else p
    out = []
    for x in rows if isinstance(rows, list) else []:
        if not isinstance(x, dict): continue
        o,h,lo,c = gf(x,"openPrice","OpenPrice"),gf(x,"highPrice","HighPrice"),gf(x,"lowPrice","LowPrice"),gf(x,"closePrice","ClosePrice")
        t = gf(x,"openTime","OpenTime")
        if None in (o,h,lo,c,t): continue
        try: o,h,lo,c = float(o),float(h),float(lo),float(c)
        except: continue
        if h<max(o,c) or lo>min(o,c) or h<lo: continue
        out.append({"time":t,"open":o,"high":h,"low":lo,"close":c})
    if not out: raise ValueError(f"{tf} 无K线")
    d = pd.DataFrame(out); d["time"] = pd.to_datetime(d.time, utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").drop_duplicates("time",keep="last").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    d = d[d.time+DELTA[tf] <= now].reset_index(drop=True)
    if len(d) < 30: raise ValueError(f"{tf} 只有{len(d)}根")
    return d

def resample(df,tf,src):
    if df is None or len(df)==0: return None
    x = df.copy(); x["b"] = x.time.dt.floor(FREQ[tf])
    a = x.groupby("b").agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),n=("open","size"))
    need = int(DELTA[tf]/DELTA[src]); now = pd.Timestamp.now(tz="UTC")
    a = a[(a.n>=max(1,int(need*0.8))) & (a.index+DELTA[tf]<=now)]
    if len(a)==0: return None
    return a.drop(columns="n").reset_index().rename(columns={"b":"time"})

def load_all():
    m5 = fetch("M5"); fr = {"M5":m5}
    for tf in ("M15","H1","H4"):
        df = None
        try:
            c = fetch(tf)
            if c is not None and len(c)>=30: df = c
        except: pass
        if df is None or len(df)<30:
            if tf=="H4" and fr.get("H1") is not None and len(fr["H1"])>=30: df = resample(fr["H1"],tf,"H1")
            else: df = resample(m5,tf,"M5")
        fr[tf] = df
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
    if s==0:
        ref = fr.get("M15")
        if ref is None or len(ref)<30: ref = fr.get("M5")
        if ref is None or len(ref)<30: return 1,0,"弱",b
        r = ind(ref).iloc[-1]; s = 1.0 if r.close>=r.EMA20 else -1.0
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
    tok, chat = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
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
    p = plan(side, fr.get("M15") or m5, price, SPREAD, BUDGET)
    strength,f = filt(side,p,strength)
    lots = max(0.01, round(p["lots"]*f,2))
    icon = "🟢" if side==1 else "🔴"; name = "BUY" if side==1 else "SELL"
    txt = (f"🥇 Kenny AI GOLD v14\n{icon} {name} LIMIT | 强度:{strength}\n"
           f"现价 {price:.2f} | RSI {p['rsi']:.1f} | ADX {p['adx']:.1f}\n"
           f"E1 {p['e1']:.2f} | E2 {p['e2']:.2f} | SL {p['sl']:.2f}\n"
           f"TP1 {p['tps'][0]:.2f} | TP2 {p['tps'][1]:.2f} | TP3 {p['tps'][2]:.2f}\n"
           f"💡 {lots}手 (预算${BUDGET})\n⚠️ 仅供参考")
    print("sent" if tg(txt) else "send failed")
