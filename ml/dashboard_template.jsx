import { useState, useEffect, useRef, useCallback } from "react";

const MARKETS=__MARKETS_JSON__;

const COORDS=__COORDS_JSON__;

const PRECOMPUTED=__PRECOMPUTED_JSON__;

const DEMOS=__DEMOS_JSON__;

const SEASONALITY=__SEASONALITY_JSON__;

const PC={"arena":"#f59e0b","theater":"#8b5cf6","club_to_theater":"#06b6d4","club":"#10b981","emerging":"#6b7280"};
const PL={"arena":"Arena","theater":"Theater","club_to_theater":"Club \u2192 Theater","club":"Club","emerging":"Emerging"};
const fmt=n=>n>=1e6?`$${(n/1e6).toFixed(1)}M`:n>=1e3?`$${(n/1e3).toFixed(0)}K`:`$${Math.round(n)}`;
const fmtN=n=>n>=1e6?`${(n/1e6).toFixed(1)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:`${Math.round(n)}`;
const pct=n=>`${Math.round(n*100)}%`;
const MONTHS=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function scoreMarkets(profile,nCities=12){
  const cap=profile.estimated_capacity||1e3,guar=profile.estimated_guarantee||1e4;
  const phase=guar>2e5?4:guar>5e4?3:guar>1e4?2:guar>2e3?1:0;
  const phaseLabel=["emerging","club","club_to_theater","theater","arena"][phase];
  const estTicket=guar>2e5?85:guar>5e4?55:guar>1e4?35:guar>2e3?25:18;
  const estMaxTicket=estTicket*1.6;
  const merchPerHead=[3,5,8,12,18][phase];
  const dealType=phase>=3?"Guarantee vs Net":phase>=2?"Guarantee Plus %":phase>=1?"Flat Guarantee":"Door Deal";
  const artistPct=phase>=4?0.85:phase>=3?0.82:phase>=2?0.80:0;
  const scored=MARKETS.map(mkt=>{
    const capRatio=mkt.mc>0?Math.min(cap,mkt.mc)/Math.max(cap,mkt.mc):0.3;
    const tierDiff=Math.abs(mkt.t-phase);
    const fillBase=mkt.af>0?mkt.af:0.5;
    const fillProb=Math.min(1,fillBase*(1-tierDiff*0.15));
    const revenueRatio=guar/Math.max(mkt.ag||5e3,1);
    const predRevenue=Math.max(0,mkt.an*Math.min(revenueRatio,3)*(0.5+fillProb*0.5));
    const predCapacity=Math.max(100,cap*(0.7+capRatio*0.6));
    const predGuarantee=mkt.ag*Math.min(revenueRatio,2.5);
    const estFill=Math.max(fillProb,0.5);
    const estSold=Math.round(predCapacity*estFill);
    const grossTicket=estSold*estTicket;
    const merchRevenue=Math.round(estSold*merchPerHead*0.85);
    const artistNet=dealType==="Flat Guarantee"?predGuarantee:Math.max(predGuarantee,grossTicket*artistPct);
    const totalIncome=artistNet+merchRevenue;
    const strengthScore=Math.min(100,mkt.na*30+mkt.ns*5);
    const score=fillProb*30+(predRevenue>0?Math.min(100,(predRevenue/Math.max(guar*1.5,1))*100)*0.25:0)+strengthScore*0.25+capRatio*20;
    return{rank:0,city:mkt.c,state:mkt.s,market:mkt.m,country:"US",
      market_score:Math.round(score*10)/10,confidence:Math.round(fillProb*100)/100,
      predicted_capacity:Math.round(predCapacity),predicted_net_revenue:Math.round(predRevenue),
      predicted_fill_probability:Math.round(fillProb*100)/100,predicted_guarantee:Math.round(predGuarantee),
      recommended_deal_type:dealType,recommended_guarantee:Math.round(predGuarantee),
      recommended_artist_pct:artistPct,deal_reasoning:"Based on estimated tier and market norms",
      profitability:{gross_ticket_revenue:Math.round(grossTicket),estimated_merch_revenue:merchRevenue,
        artist_net:Math.round(artistNet),total_artist_income:Math.round(totalIncome),
        recommended_avg_ticket_price:estTicket,ticket_price_range:[Math.round(estTicket*0.6),Math.round(estMaxTicket)],
        est_sold:estSold,est_fill_rate:Math.round(estFill*100)/100,n_tiers:mkt.nt||3,merch_per_head:merchPerHead},
      recommended_avg_ticket_price:estTicket,ticket_price_range:[Math.round(estTicket*0.6),Math.round(estMaxTicket)],
      best_months:["Mar","Apr","May","Sep","Oct","Nov"],seasonality_score:0.1,
      has_artist_history:false,historical:null,
      reasoning:[fillProb>.7?`High fill probability (${pct(fillProb)})`:null,mkt.na>=2?`Strong market (${mkt.na} artists)`:null,capRatio>.6?"Good capacity match":null].filter(Boolean).join(". ")+"."||"Recommended by model."};
  }).filter(m=>m.market_score>10).sort((a,b)=>b.market_score-a.market_score).slice(0,nCities);
  scored.forEach((m,i)=>m.rank=i+1);
  const getC=m=>COORDS[m]||[39.8,-98.6];
  const dist=(a,b)=>Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)*69;
  const mkts=scored.map(m=>m.market);
  if(mkts.length>2){let imp=true;while(imp){imp=false;for(let i=1;i<mkts.length-1;i++)for(let j=i+1;j<mkts.length;j++){const dO=dist(getC(mkts[i-1]),getC(mkts[i]))+dist(getC(mkts[j]),getC(mkts[(j+1)%mkts.length]));const dN=dist(getC(mkts[i-1]),getC(mkts[j]))+dist(getC(mkts[i]),getC(mkts[(j+1)%mkts.length]));if(dN<dO){const s=mkts.slice(i,j+1).reverse();mkts.splice(i,j-i+1,...s);imp=true}}}}
  let totalD=0;for(let i=0;i<mkts.length-1;i++)totalD+=dist(getC(mkts[i]),getC(mkts[i+1]));
  const totalNet=scored.reduce((s,m)=>s+(m.profitability?.artist_net||m.predicted_net_revenue),0);
  const totalMerch=scored.reduce((s,m)=>s+(m.profitability?.estimated_merch_revenue||0),0);
  const totalIncome2=scored.reduce((s,m)=>s+(m.profitability?.total_artist_income||m.predicted_net_revenue),0);
  const avgGuar=scored.reduce((s,m)=>s+m.recommended_guarantee,0)/Math.max(scored.length,1);
  const avgTicket=scored.reduce((s,m)=>s+(m.recommended_avg_ticket_price||0),0)/Math.max(scored.length,1);
  const dealCounts={};scored.forEach(m=>{dealCounts[m.recommended_deal_type]=(dealCounts[m.recommended_deal_type]||0)+1});
  return{artist:profile.name||"Unknown",profile_source:"ai_analysis",
    artist_profile:{growth_phase:phaseLabel,current_avg_capacity:cap,current_avg_guarantee:guar,avg_fill_rate:0.7,guarantee_cagr:0,total_headline_shows:0,n_markets_played:0,avg_ticket_price:estTicket,avg_merch_soft_pct:0.85},
    recommendation_config:{optimize_for:"balanced",region:"us",n_requested:nCities},
    markets:scored,route:{order:mkts.map((_,i)=>i),routed_markets:mkts,total_distance_miles:Math.round(totalD),start_city:mkts[0]},
    financial_summary:{total_predicted_net:Math.round(totalNet),total_predicted_merch:Math.round(totalMerch),total_predicted_income:Math.round(totalIncome2),avg_recommended_guarantee:Math.round(avgGuar),avg_ticket_price:Math.round(avgTicket*100)/100,avg_predicted_capacity:Math.round(scored.reduce((s,m)=>s+m.predicted_capacity,0)/Math.max(scored.length,1)),avg_predicted_fill_prob:Math.round(scored.reduce((s,m)=>s+m.predicted_fill_probability,0)/Math.max(scored.length,1)*100)/100,deal_type_breakdown:dealCounts,best_tour_months:["Mar","Apr","May","Sep","Oct","Nov"]},
    model_metadata:{model_type:"cross_artist_gradient_boosted",n_features:__N_FEATURES__,fill_auc:__FILL_AUC__,revenue_r2:__REVENUE_R2__,n_training_shows:__N_SHOWS__,n_training_artists:14,n_markets:__N_MARKETS__,guarantee_r2:__GUARANTEE_R2__,top_features:{}}};
}

function MapViz({data}){
  const canvasRef=useRef(null);const[hovered,setHovered]=useState(null);const[dims,setDims]=useState({w:800,h:400});const containerRef=useRef(null);
  useEffect(()=>{const obs=new ResizeObserver(e=>{for(let en of e)setDims({w:en.contentRect.width,h:Math.max(300,en.contentRect.width*.48)})});if(containerRef.current)obs.observe(containerRef.current);return()=>obs.disconnect()},[]);
  const project=useCallback((lat,lon)=>{const p=40;return[p+((lon-(-125))/((-66)-(-125)))*(dims.w-p*2),p+((50-lat)/(50-24))*(dims.h-p*2)]},[dims]);
  useEffect(()=>{const canvas=canvasRef.current;if(!canvas)return;const ctx=canvas.getContext("2d");const dpr=window.devicePixelRatio||1;canvas.width=dims.w*dpr;canvas.height=dims.h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,dims.w,dims.h);
    const route=data.route.routed_markets;ctx.strokeStyle="rgba(245,158,11,0.2)";ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.beginPath();route.forEach((m,i)=>{const c=COORDS[m];if(!c)return;const[x,y]=project(c[0],c[1]);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.stroke();ctx.setLineDash([]);
    data.markets.forEach((m,i)=>{const c=COORDS[m.market];if(!c)return;const[x,y]=project(c[0],c[1]);const maxC=data.artist_profile.current_avg_capacity||1e3;const r=Math.max(5,Math.min(18,m.predicted_capacity/maxC*8));const isH=hovered===i;
      if(isH){const g=ctx.createRadialGradient(x,y,0,x,y,r*3);g.addColorStop(0,"rgba(245,158,11,0.3)");g.addColorStop(1,"rgba(245,158,11,0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r*3,0,Math.PI*2);ctx.fill()}
      const a=.4+m.confidence*.6;ctx.fillStyle=m.has_artist_history?`rgba(245,158,11,${a})`:`rgba(6,182,212,${a})`;ctx.beginPath();ctx.arc(x,y,isH?r*1.4:r,0,Math.PI*2);ctx.fill();ctx.strokeStyle=isH?"#fff":"rgba(255,255,255,0.4)";ctx.lineWidth=isH?2:1;ctx.stroke();
      ctx.fillStyle="#fff";ctx.font=isH?"bold 11px 'DM Sans',sans-serif":"10px 'DM Sans',sans-serif";ctx.textAlign="center";ctx.fillText(m.city,x,y-r-6);
      const ri=route.indexOf(m.market);if(ri>=0){ctx.fillStyle="rgba(0,0,0,0.7)";ctx.font="bold 9px 'DM Sans',sans-serif";ctx.textBaseline="middle";ctx.fillText(ri+1,x,y);ctx.textBaseline="alphabetic"}})
  },[data,hovered,dims,project]);
  const handleMove=e=>{const rect=canvasRef.current.getBoundingClientRect();const mx=e.clientX-rect.left,my=e.clientY-rect.top;let found=null;data.markets.forEach((m,i)=>{const c=COORDS[m.market];if(!c)return;const[x,y]=project(c[0],c[1]);if(Math.hypot(mx-x,my-y)<20)found=i});setHovered(found)};
  return(<div ref={containerRef} style={{position:"relative",width:"100%"}}><canvas ref={canvasRef} style={{width:"100%",height:dims.h,cursor:hovered!==null?"pointer":"default"}} onMouseMove={handleMove} onMouseLeave={()=>setHovered(null)}/>
    {hovered!==null&&(()=>{const m=data.markets[hovered];const c=COORDS[m.market];if(!c)return null;const[x,y]=project(c[0],c[1]);return(<div style={{position:"absolute",left:Math.min(x,dims.w-240),top:y+20,background:"rgba(15,15,20,0.95)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:8,padding:"10px 14px",width:220,pointerEvents:"none",zIndex:10,backdropFilter:"blur(10px)"}}><div style={{fontWeight:700,fontSize:13,color:"#f59e0b"}}>{m.market}</div><div style={{fontSize:11,color:"#94a3b8",marginTop:4,lineHeight:1.6}}>Score: {m.market_score} &middot; Fill: {pct(m.predicted_fill_probability)}<br/>Capacity: {fmtN(m.predicted_capacity)} &middot; Revenue: {fmt(m.predicted_net_revenue)}<br/>{m.profitability?`Income: ${fmt(m.profitability.total_artist_income)}`:""}{m.has_artist_history?` \u00b7 ${m.historical?.prior_visits||0} visits`:" \u00b7 New market"}</div></div>)})()}</div>);
}

function ScoreBar({value,max=100,color="#f59e0b"}){return(<div style={{width:"100%",height:4,background:"rgba(255,255,255,0.08)",borderRadius:2}}><div style={{width:`${(value/max)*100}%`,height:"100%",background:color,borderRadius:2,transition:"width 0.6s ease"}}/></div>)}

function MarketCard({market,index}){const[open,setOpen]=useState(false);const p=market.profitability;return(
  <div onClick={()=>setOpen(!open)} style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:8,padding:"12px 14px",cursor:"pointer",transition:"all 0.2s",borderColor:open?"rgba(245,158,11,0.3)":undefined}}>
    <div style={{display:"flex",alignItems:"center",gap:10}}>
      <div style={{width:28,height:28,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",background:market.has_artist_history?"rgba(245,158,11,0.15)":"rgba(6,182,212,0.15)",color:market.has_artist_history?"#f59e0b":"#06b6d4",fontSize:12,fontWeight:800,flexShrink:0}}>{index+1}</div>
      <div style={{flex:1,minWidth:0}}><div style={{fontWeight:700,fontSize:13,color:"#e2e8f0"}}>{market.city}, {market.state}</div><div style={{fontSize:10,color:"#64748b",marginTop:1}}>{market.has_artist_history?`${market.historical?.prior_visits||0} visits`:"New"}</div></div>
      <div style={{textAlign:"right",flexShrink:0}}><div style={{fontSize:16,fontWeight:800,color:"#f59e0b",fontVariantNumeric:"tabular-nums"}}>{market.market_score.toFixed(1)}</div><div style={{fontSize:9,color:"#64748b",textTransform:"uppercase",letterSpacing:.5}}>score</div></div>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginTop:10}}>
      {[{label:"Fill",value:pct(market.predicted_fill_probability),pct:market.predicted_fill_probability},{label:"Cap",value:fmtN(market.predicted_capacity),pct:Math.min(market.predicted_capacity/20000,1)},{label:"Income",value:fmt(p?.total_artist_income||market.predicted_net_revenue),pct:Math.min((p?.total_artist_income||market.predicted_net_revenue)/500000,1)}].map(s=>(
        <div key={s.label} style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:2,alignItems:"baseline"}}><span style={{fontSize:9,color:"#64748b",textTransform:"uppercase",letterSpacing:.3}}>{s.label}</span><span style={{fontSize:10,fontWeight:600,color:"#cbd5e1",fontVariantNumeric:"tabular-nums"}}>{s.value}</span></div><ScoreBar value={s.pct*100}/></div>
      ))}
    </div>
    {open&&(<div style={{marginTop:10,paddingTop:10,borderTop:"1px solid rgba(255,255,255,0.06)",fontSize:11,color:"#94a3b8",lineHeight:1.6}}>
      <div style={{fontStyle:"italic",marginBottom:8}}>{market.reasoning}</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"4px 12px",fontSize:10}}>
        {market.recommended_deal_type&&<><span style={{color:"#64748b"}}>Deal: <b style={{color:"#f59e0b"}}>{market.recommended_deal_type}</b></span><span style={{color:"#64748b"}}>Guarantee: <b style={{color:"#cbd5e1"}}>{fmt(market.recommended_guarantee||0)}</b></span></>}
        {market.recommended_artist_pct>0&&<span style={{color:"#64748b"}}>Artist %: <b style={{color:"#cbd5e1"}}>{pct(market.recommended_artist_pct)}</b></span>}
        {market.recommended_avg_ticket_price>0&&<span style={{color:"#64748b"}}>Ticket: <b style={{color:"#cbd5e1"}}>${market.recommended_avg_ticket_price.toFixed(0)}</b></span>}
        {p&&<><span style={{color:"#64748b"}}>Gross: <b style={{color:"#cbd5e1"}}>{fmt(p.gross_ticket_revenue||0)}</b></span><span style={{color:"#64748b"}}>Merch: <b style={{color:"#10b981"}}>{fmt(p.estimated_merch_revenue||0)}</b></span><span style={{color:"#64748b"}}>Net: <b style={{color:"#cbd5e1"}}>{fmt(p.artist_net||0)}</b></span><span style={{color:"#64748b"}}>Total: <b style={{color:"#f59e0b"}}>{fmt(p.total_artist_income||0)}</b></span></>}
        {market.best_months&&<span style={{color:"#64748b",gridColumn:"1/-1"}}>Best months: <b style={{color:"#06b6d4"}}>{market.best_months?.join(", ")}</b></span>}
      </div>
      {market.has_artist_history&&market.historical&&(<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"3px 12px",fontSize:10,color:"#64748b",marginTop:5}}><span>Prior fill: <b style={{color:"#cbd5e1"}}>{pct(market.historical.prior_avg_fill_rate||0)}</b></span><span>Last cap: <b style={{color:"#cbd5e1"}}>{fmtN(market.historical.prior_last_capacity||0)}</b></span></div>)}
    </div>)}
  </div>)}

function ProfitabilityView({data}){
  const fs=data.financial_summary;const mkts=data.markets;
  const maxIncome=Math.max(...mkts.map(m=>m.profitability?.total_artist_income||m.predicted_net_revenue||1));
  return(<div style={{display:"flex",flexDirection:"column",gap:16}}>
    <div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",padding:"14px 16px"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#f59e0b",marginBottom:12}}>Tour P&L Summary</div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
          <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
            <th style={{textAlign:"left",padding:"6px 8px",color:"#64748b",fontWeight:600,fontSize:9,textTransform:"uppercase",letterSpacing:.5}}>Category</th>
            <th style={{textAlign:"right",padding:"6px 8px",color:"#64748b",fontWeight:600,fontSize:9,textTransform:"uppercase",letterSpacing:.5}}>Per Show Avg</th>
            <th style={{textAlign:"right",padding:"6px 8px",color:"#64748b",fontWeight:600,fontSize:9,textTransform:"uppercase",letterSpacing:.5}}>Tour Total</th>
          </tr></thead>
          <tbody>
            {[["Gross Ticket Revenue",mkts.reduce((s,m)=>s+(m.profitability?.gross_ticket_revenue||0),0)],
              ["Artist Net",fs.total_predicted_net],
              ["Merch Revenue",fs.total_predicted_merch||0],
              ["Total Income",fs.total_predicted_income||fs.total_predicted_net]].map(([label,total])=>(
              <tr key={label} style={{borderBottom:"1px solid rgba(255,255,255,0.04)"}}><td style={{padding:"6px 8px",color:label==="Total Income"?"#f59e0b":"#e2e8f0",fontWeight:label==="Total Income"?700:400}}>{label}</td><td style={{textAlign:"right",padding:"6px 8px",color:"#cbd5e1",fontFamily:"'JetBrains Mono',monospace"}}>{fmt(Math.round(total/Math.max(mkts.length,1)))}</td><td style={{textAlign:"right",padding:"6px 8px",color:label==="Total Income"?"#f59e0b":"#cbd5e1",fontWeight:label==="Total Income"?700:400,fontFamily:"'JetBrains Mono',monospace"}}>{fmt(Math.round(total))}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      {fs.deal_type_breakdown&&<div style={{marginTop:10,display:"flex",gap:6,flexWrap:"wrap"}}>
        {Object.entries(fs.deal_type_breakdown).map(([type,count])=>(
          <span key={type} style={{padding:"3px 8px",borderRadius:12,fontSize:9,fontWeight:600,background:"rgba(139,92,246,0.1)",color:"#a78bfa",border:"1px solid rgba(139,92,246,0.2)"}}>{type} ({count})</span>
        ))}
      </div>}
    </div>
    <div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",padding:"14px 16px"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#f59e0b",marginBottom:12}}>Per-Market Income Breakdown</div>
      <div style={{display:"flex",flexDirection:"column",gap:6}}>
        {mkts.map((m,i)=>{const p=m.profitability;const net=p?.artist_net||m.predicted_net_revenue||0;const merch=p?.estimated_merch_revenue||0;const total=p?.total_artist_income||net;const guar=m.recommended_guarantee||0;return(
          <div key={m.market} style={{display:"flex",alignItems:"center",gap:8}}>
            <div style={{width:100,fontSize:10,color:"#94a3b8",flexShrink:0,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{m.city}</div>
            <div style={{flex:1,position:"relative",height:20,background:"rgba(255,255,255,0.03)",borderRadius:3}}>
              <div style={{position:"absolute",left:0,top:0,height:"100%",width:`${(net/maxIncome)*100}%`,background:"rgba(245,158,11,0.6)",borderRadius:"3px 0 0 3px"}}/>
              <div style={{position:"absolute",left:`${(net/maxIncome)*100}%`,top:0,height:"100%",width:`${(merch/maxIncome)*100}%`,background:"rgba(16,185,129,0.6)"}}/>
              {guar>0&&<div style={{position:"absolute",left:`${(guar/maxIncome)*100}%`,top:0,height:"100%",width:2,background:"#f59e0b"}}/>}
            </div>
            <div style={{width:60,fontSize:10,color:"#cbd5e1",textAlign:"right",fontFamily:"'JetBrains Mono',monospace",flexShrink:0}}>{fmt(total)}</div>
          </div>
        )})}
      </div>
      <div style={{display:"flex",gap:12,marginTop:10,fontSize:9,color:"#64748b"}}>
        <span><span style={{display:"inline-block",width:8,height:8,borderRadius:2,background:"rgba(245,158,11,0.6)",marginRight:4}}/>Artist Net</span>
        <span><span style={{display:"inline-block",width:8,height:8,borderRadius:2,background:"rgba(16,185,129,0.6)",marginRight:4}}/>Merch</span>
        <span><span style={{display:"inline-block",width:8,height:2,background:"#f59e0b",marginRight:4,verticalAlign:"middle"}}/>Guarantee</span>
      </div>
    </div>
  </div>);
}

function TimingView({data,artistKey}){
  const slug=artistKey||data.artist?.toLowerCase().replace(/\s+/g,"_").replace(/'/g,"");
  const seasonal=SEASONALITY[slug];
  if(!seasonal||seasonal.length===0)return(<div style={{padding:20,textAlign:"center",color:"#64748b",fontSize:12}}>No seasonality data available for this artist.</div>);
  const maxFill=Math.max(...seasonal.filter(m=>m.n>0).map(m=>m.f),0.01);
  const maxRev=Math.max(...seasonal.filter(m=>m.n>0).map(m=>m.r),1);
  const bestMonths=data.financial_summary?.best_tour_months||["Mar","Apr","May","Sep","Oct","Nov"];
  return(<div style={{display:"flex",flexDirection:"column",gap:16}}>
    <div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",padding:"14px 16px"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#f59e0b",marginBottom:4}}>Monthly Fill Rate</div>
      <div style={{fontSize:10,color:"#64748b",marginBottom:12}}>Based on historical headline shows</div>
      <div style={{display:"flex",alignItems:"flex-end",gap:4,height:120}}>
        {seasonal.map((m,i)=>{const isBest=bestMonths.includes(MONTHS[m.m]);return(
          <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
            <div style={{fontSize:9,color:"#cbd5e1",fontFamily:"'JetBrains Mono',monospace"}}>{m.n>0?pct(m.f):""}</div>
            <div style={{width:"100%",background:m.n>0?(isBest?"rgba(245,158,11,0.5)":"rgba(100,116,139,0.3)"):"rgba(255,255,255,0.03)",borderRadius:3,height:`${m.n>0?(m.f/maxFill)*100:5}%`,minHeight:4,transition:"height 0.4s ease"}}/>
            <div style={{fontSize:9,color:isBest?"#f59e0b":"#64748b",fontWeight:isBest?700:400}}>{MONTHS[m.m]}</div>
            <div style={{fontSize:8,color:"#475569"}}>{m.n>0?`${m.n}`:"-"}</div>
          </div>
        )})}
      </div>
    </div>
    <div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",padding:"14px 16px"}}>
      <div style={{fontSize:12,fontWeight:700,color:"#f59e0b",marginBottom:4}}>Monthly Revenue</div>
      <div style={{fontSize:10,color:"#64748b",marginBottom:12}}>Average artist net by month</div>
      <div style={{display:"flex",alignItems:"flex-end",gap:4,height:120}}>
        {seasonal.map((m,i)=>{const isBest=bestMonths.includes(MONTHS[m.m]);return(
          <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
            <div style={{fontSize:9,color:"#cbd5e1",fontFamily:"'JetBrains Mono',monospace"}}>{m.n>0?fmt(m.r):""}</div>
            <div style={{width:"100%",background:m.n>0?(isBest?"rgba(6,182,212,0.5)":"rgba(100,116,139,0.3)"):"rgba(255,255,255,0.03)",borderRadius:3,height:`${m.n>0?(m.r/maxRev)*100:5}%`,minHeight:4,transition:"height 0.4s ease"}}/>
            <div style={{fontSize:9,color:isBest?"#06b6d4":"#64748b",fontWeight:isBest?700:400}}>{MONTHS[m.m]}</div>
          </div>
        )})}
      </div>
    </div>
    <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
      <div style={{padding:"6px 10px",borderRadius:6,fontSize:10,background:"rgba(245,158,11,0.08)",color:"#f59e0b",border:"1px solid rgba(245,158,11,0.2)"}}>Best: {bestMonths.join(", ")}</div>
      <div style={{padding:"6px 10px",borderRadius:6,fontSize:10,background:"rgba(100,116,139,0.08)",color:"#94a3b8",border:"1px solid rgba(100,116,139,0.2)"}}>Avoid: Jul, Aug (festival competition)</div>
    </div>
  </div>);
}

export default function App(){
  const[data,setData]=useState(PRECOMPUTED[DEMOS[0]?.key]||Object.values(PRECOMPUTED)[0]);const[activeDemo,setActiveDemo]=useState(DEMOS[0]?.key);const[searchQuery,setSearchQuery]=useState("");const[loading,setLoading]=useState(false);const[error,setError]=useState(null);const[view,setView]=useState("map");const[sidebarOpen,setSidebarOpen]=useState(false);
  const ap=data.artist_profile;const phase=ap.growth_phase;const fs=data.financial_summary;

  const handleSearch=async()=>{const q=searchQuery.trim();if(!q)return;setLoading(true);setError(null);setActiveDemo(null);
    try{const res=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({artist:q})});
      const result=await res.json();const text=result.content?.map(b=>b.text||"").join("")||"";const clean=text.replace(/```json|```/g,"").trim();const profile=JSON.parse(clean);setData(scoreMarkets(profile));
    }catch(err){setError("Could not analyze artist. Try again or check the name.");console.error(err)}finally{setLoading(false)}};

  return(
    <div style={{minHeight:"100vh",background:"#0a0a0f",color:"#e2e8f0",fontFamily:"'DM Sans','Helvetica Neue',sans-serif"}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;0,9..40,800;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}} @media(max-width:768px){.hide-mobile{display:none!important}}`}</style>

      {/* Header */}
      <div style={{padding:"16px",borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
          <button onClick={()=>setSidebarOpen(!sidebarOpen)} style={{padding:"8px 12px",borderRadius:6,border:"1px solid rgba(255,255,255,0.1)",background:"rgba(255,255,255,0.05)",color:"#f59e0b",fontSize:20,cursor:"pointer",lineHeight:1}}>{"\u2630"}</button>
          <div style={{flex:1,minWidth:0}}><div style={{fontSize:10,fontWeight:700,letterSpacing:2,color:"#f59e0b",textTransform:"uppercase"}}>Wilder AI</div><div style={{fontSize:16,fontWeight:800,color:"#f8fafc",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>Tour Recommendation</div></div>
        </div>
        <div style={{display:"flex",gap:8,marginBottom:8}}>
          <input value={searchQuery} onChange={e=>setSearchQuery(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")handleSearch()}} placeholder="Search any artist..." style={{flex:1,padding:"10px 14px",borderRadius:8,border:"1px solid rgba(255,255,255,0.12)",background:"rgba(255,255,255,0.05)",color:"#f8fafc",fontSize:14,outline:"none",fontFamily:"inherit",boxSizing:"border-box"}}/>
          <button onClick={handleSearch} disabled={loading||!searchQuery.trim()} style={{padding:"10px 16px",borderRadius:8,border:"none",cursor:loading?"wait":"pointer",background:loading?"rgba(245,158,11,0.3)":"#f59e0b",color:"#0a0a0f",fontSize:13,fontWeight:700,fontFamily:"inherit",opacity:!searchQuery.trim()?0.4:1,whiteSpace:"nowrap"}}>
            {loading?<span style={{display:"inline-block",width:14,height:14,border:"2px solid rgba(0,0,0,0.2)",borderTopColor:"#0a0a0f",borderRadius:"50%",animation:"spin .8s linear infinite"}}/>:"Go"}
          </button>
        </div>
        <div style={{padding:"6px 10px",borderRadius:6,fontSize:10,fontWeight:600,background:"rgba(245,158,11,0.1)",color:"#f59e0b",border:"1px solid rgba(245,158,11,0.2)",display:"inline-block"}}>{data.model_metadata?.n_training_shows||__N_SHOWS__} shows &middot; {data.model_metadata?.n_markets||__N_MARKETS__} markets &middot; {data.model_metadata?.n_features||__N_FEATURES__} features</div>
      </div>

      <div style={{display:"flex",gap:0,minHeight:"calc(100vh - 140px)",position:"relative"}}>
        {/* Sidebar */}
        <div style={{width:sidebarOpen?240:0,borderRight:sidebarOpen?"1px solid rgba(255,255,255,0.06)":"none",padding:sidebarOpen?"16px 12px":0,flexShrink:0,transition:"all 0.3s",overflow:sidebarOpen?"visible":"hidden",position:"absolute",left:0,top:0,bottom:0,background:"#0a0a0f",zIndex:100}}>
          <div style={{fontSize:9,fontWeight:700,letterSpacing:1.5,color:"#64748b",textTransform:"uppercase",marginBottom:8}}>Demo Artists</div>
          {DEMOS.map(s=>(
            <div key={s.key} onClick={()=>{setActiveDemo(s.key);setData(PRECOMPUTED[s.key]);setSearchQuery("");setError(null);setSidebarOpen(false)}} style={{padding:"8px 10px",borderRadius:6,cursor:"pointer",marginBottom:4,background:activeDemo===s.key?"rgba(245,158,11,0.08)":"transparent",border:`1px solid ${activeDemo===s.key?"rgba(245,158,11,0.2)":"transparent"}`,transition:"all 0.15s"}}>
              <div style={{display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:16}}>{s.icon}</span><div><div style={{fontSize:12,fontWeight:700,color:activeDemo===s.key?"#f59e0b":"#e2e8f0"}}>{s.label}</div><div style={{fontSize:9,color:"#64748b"}}>{s.sub}</div></div></div>
            </div>))}

          <div style={{marginTop:16,padding:"10px",background:"rgba(255,255,255,0.02)",borderRadius:6,border:"1px solid rgba(255,255,255,0.05)"}}>
            <div style={{fontSize:9,fontWeight:700,letterSpacing:1.5,color:"#64748b",textTransform:"uppercase",marginBottom:6}}>Artist Profile</div>
            <div style={{fontSize:14,fontWeight:800,color:"#f8fafc",marginBottom:4}}>{data.artist}</div>
            <div style={{display:"inline-block",padding:"2px 8px",borderRadius:12,fontSize:10,fontWeight:700,background:`${PC[phase]||"#6b7280"}20`,color:PC[phase]||"#6b7280",border:`1px solid ${PC[phase]||"#6b7280"}40`,marginBottom:6}}>{PL[phase]||phase}</div>
            <div style={{fontSize:10,color:"#94a3b8",lineHeight:1.8}}>
              <div>Source: <b style={{color:"#cbd5e1"}}>{data.profile_source?.replace(/_/g," ")}</b></div>
              <div>Cap: <b style={{color:"#cbd5e1"}}>{fmtN(ap.current_avg_capacity)}</b></div>
              <div>Guar: <b style={{color:"#cbd5e1"}}>{fmt(ap.current_avg_guarantee)}</b></div>
              <div>Fill: <b style={{color:"#cbd5e1"}}>{pct(ap.avg_fill_rate)}</b></div>
              {ap.avg_ticket_price>0&&<div>Ticket: <b style={{color:"#cbd5e1"}}>${ap.avg_ticket_price?.toFixed(0)}</b></div>}
              {ap.total_headline_shows>0&&<div>Shows: <b style={{color:"#cbd5e1"}}>{ap.total_headline_shows}</b></div>}
            </div>
            {data.ai_summary&&(<div style={{marginTop:8,padding:"6px 8px",background:"rgba(6,182,212,0.06)",borderRadius:4,border:"1px solid rgba(6,182,212,0.15)",fontSize:10,color:"#94a3b8",lineHeight:1.5}}><div style={{fontSize:9,fontWeight:700,color:"#06b6d4",marginBottom:3}}>AI ANALYSIS</div>{data.ai_summary}</div>)}
          </div>
        </div>

        {/* Backdrop */}
        {sidebarOpen&&<div onClick={()=>setSidebarOpen(false)} style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.5)",zIndex:99}}/>}

        {/* Main */}
        <div style={{flex:1,padding:"16px",overflow:"auto",width:"100%"}}>
          {error&&(<div style={{padding:"12px 16px",background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.3)",borderRadius:8,color:"#fca5a5",fontSize:13,marginBottom:16}}>{error}</div>)}

          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(140px,1fr))",gap:10,marginBottom:16}}>
            {[{label:"Total Income",value:fmt(fs.total_predicted_income||fs.total_predicted_net),sub:"net + merch"},{label:"Avg Guarantee",value:fmt(fs.avg_recommended_guarantee||ap.current_avg_guarantee||0),sub:"recommended"},{label:"Avg Capacity",value:fmtN(fs.avg_predicted_capacity),sub:"predicted"},{label:"Avg Fill",value:pct(fs.avg_predicted_fill_prob),sub:"probability"},{label:"Avg Ticket",value:`$${(fs.avg_ticket_price||ap.avg_ticket_price||0).toFixed(0)}`,sub:"recommended"},{label:"Route",value:`${fmtN(data.route.total_distance_miles)} mi`,sub:`${data.route.routed_markets.length} stops`}].map(s=>(
              <div key={s.label} style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:8,padding:"12px 14px"}}><div style={{fontSize:9,fontWeight:700,letterSpacing:1,color:"#64748b",textTransform:"uppercase"}}>{s.label}</div><div style={{fontSize:18,fontWeight:800,color:"#f8fafc",marginTop:2,fontVariantNumeric:"tabular-nums",fontFamily:"'JetBrains Mono',monospace"}}>{s.value}</div><div style={{fontSize:10,color:"#475569",marginTop:1}}>{s.sub}</div></div>
            ))}
          </div>

          <div style={{display:"flex",gap:4,marginBottom:12,overflowX:"auto"}}>
            {[{k:"map",l:"Map"},{k:"markets",l:"Markets"},{k:"profitability",l:"Profitability"},{k:"route",l:"Route"},{k:"timing",l:"Timing"}].map(t=>(<button key={t.k} onClick={()=>setView(t.k)} style={{padding:"8px 14px",borderRadius:6,border:"1px solid",borderColor:view===t.k?"rgba(245,158,11,0.3)":"rgba(255,255,255,0.08)",background:view===t.k?"rgba(245,158,11,0.08)":"transparent",color:view===t.k?"#f59e0b":"#94a3b8",fontSize:11,fontWeight:600,cursor:"pointer",whiteSpace:"nowrap",flexShrink:0}}>{t.l}</button>))}
          </div>

          {view==="map"&&(<div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",overflow:"hidden"}}><MapViz data={data}/><div style={{padding:"10px 14px",borderTop:"1px solid rgba(255,255,255,0.06)",display:"flex",gap:12,flexWrap:"wrap",fontSize:10,color:"#64748b"}}><span><span style={{display:"inline-block",width:8,height:8,borderRadius:"50%",background:"#f59e0b",marginRight:4}}/>Known</span><span><span style={{display:"inline-block",width:8,height:8,borderRadius:"50%",background:"#06b6d4",marginRight:4}}/>New</span></div></div>)}

          {view==="markets"&&(<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))",gap:10}}>{data.markets.map((m,i)=><MarketCard key={m.market} market={m} index={i}/>)}</div>)}

          {view==="profitability"&&<ProfitabilityView data={data}/>}

          {view==="route"&&(<div style={{background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)",padding:"14px 16px"}}><div style={{fontSize:12,fontWeight:700,color:"#f59e0b",marginBottom:12}}>Optimized Route &middot; {data.route.total_distance_miles.toLocaleString()} mi</div>
            {data.route.routed_markets.map((mkt,i)=>{const m=data.markets.find(x=>x.market===mkt);return(<div key={mkt} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 0",borderBottom:i<data.route.routed_markets.length-1?"1px solid rgba(255,255,255,0.04)":"none"}}><div style={{width:24,height:24,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(245,158,11,0.1)",color:"#f59e0b",fontSize:11,fontWeight:800,flexShrink:0}}>{i+1}</div><div style={{flex:1,minWidth:0}}><div style={{fontWeight:700,color:"#e2e8f0",fontSize:13}}>{mkt}</div>{m&&<div style={{color:"#64748b",fontSize:10,marginTop:1}}>Score {m.market_score} &middot; {fmtN(m.predicted_capacity)} &middot; {fmt(m.profitability?.total_artist_income||m.predicted_net_revenue)} income</div>}</div>{i<data.route.routed_markets.length-1&&<div style={{color:"#334155",fontSize:16,flexShrink:0}}>{"\u2192"}</div>}</div>)})}</div>)}

          {view==="timing"&&<TimingView data={data} artistKey={activeDemo}/>}

          <div style={{marginTop:16,padding:"12px 14px",background:"rgba(255,255,255,0.02)",borderRadius:10,border:"1px solid rgba(255,255,255,0.06)"}}><div style={{fontSize:9,fontWeight:700,letterSpacing:1.5,color:"#64748b",textTransform:"uppercase",marginBottom:10}}>Model Features</div><div style={{display:"flex",gap:6,flexWrap:"wrap"}}>{Object.entries(data.model_metadata?.top_features||{}).map(([k,v])=>(<div key={k} style={{padding:"4px 8px",borderRadius:5,fontSize:10,fontFamily:"'JetBrains Mono',monospace",background:`rgba(245,158,11,${.05+v*.5})`,color:"#f59e0b",border:"1px solid rgba(245,158,11,0.15)"}}>{k.replace(/_/g," ")} <b>{(v*100).toFixed(0)}%</b></div>))}</div></div>
        </div>
      </div>
    </div>);
}
