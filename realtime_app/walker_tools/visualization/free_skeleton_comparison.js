/* Frame-indexed saved results. No state estimation, interpolation or pose edits. */
(async()=>{
  const response=await fetch('data.json.gz');if(!response.ok)throw Error(`数据请求 ${response.status}`);
  const json=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).text();
  const d=JSON.parse(json),unpack=s=>new Float32Array(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer);
  for(const m of d.models)m.xyz=unpack(m.points);
  const vertices=unpack(d.mesh),raw=unpack(d.raw),walker=unpack(d.walker),cameras=unpack(d.cameras);
  const $=id=>document.getElementById(id),canvas=$('scene'),ctx=canvas.getContext('2d');
  const E=[[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  const valid=p=>p.every(Number.isFinite),point=(a,j,frame=fi,n=17)=>Array.from(a.subarray((frame*n+j)*3,(frame*n+j)*3+3));
  const median=a=>{const b=a.filter(x=>x!==null&&Number.isFinite(x)).sort((x,y)=>x-y);return b.length?(b[Math.floor((b.length-1)/2)]+b[Math.ceil((b.length-1)/2)])/2:null;};
  const number=(v,n=1)=>v===null||!Number.isFinite(v)?'—':v.toFixed(n);
  const states={initialized:'初始化',updated:'更新',predicted:'仅预测',reinitialized:'重启',expired:'失效',uninitialized:'缺失',feasible_unconverged:'未收敛',converged:'收敛',not_run:'首帧'};
  let fi=61,selected=0,yaw=.65,pitch=.12,zoom=.85,panX=0,panY=0,playing=false,lastTime=0,drawSerial=0;
  const requested=Number(new URLSearchParams(location.search).get('frame'));if(new URLSearchParams(location.search).has('frame')&&Number.isInteger(requested)&&requested>=0&&requested<448)fi=requested;
  for(let i=0;i<d.models.length;i++){const o=document.createElement('option');o.value=i;o.textContent=d.models[i].label;$('variant').append(o);}
  function project(p){
    const x=p[0]-d.center[0],y=-(p[1]-d.center[1]),z=p[2]-.85;
    const xx=x*Math.cos(yaw)-y*Math.sin(yaw),yy=x*Math.sin(yaw)+y*Math.cos(yaw);
    const up=z*Math.cos(pitch)-yy*Math.sin(pitch),depth=z*Math.sin(pitch)+yy*Math.cos(pitch);
    const s=Math.min(canvas.width,canvas.height)*.85*zoom/(2+depth*.15);
    return [canvas.width*.5+panX+xx*s,canvas.height*.53+panY-up*s,depth];
  }
  function line(a,b,color,width=2,dash=[]){if(!valid(a)||!valid(b))return;const A=project(a),B=project(b);ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.setLineDash(dash.map(x=>x*devicePixelRatio));ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();ctx.setLineDash([]);}
  function dot(p,color,r=3){if(!valid(p))return;const q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*devicePixelRatio,0,Math.PI*2);ctx.fill();}
  function skeleton(model,reference=false){
    const statuses=model.status[fi];let edges=0;
    for(const [a,b] of E){const A=point(model.xyz,a),B=point(model.xyz,b);if(!valid(A)||!valid(B))continue;
      const predicted=(a>=5&&statuses[a-5]==='predicted')||(b>=5&&statuses[b-5]==='predicted');
      line(A,B,model.color,reference?2:3,predicted?[4,3]:[]);edges++;}
    for(let j=5;j<17;j++)dot(point(model.xyz,j),model.color,reference?2:3.4);
    return edges;
  }
  function drawScene(){
    canvas.width=Math.round(canvas.clientWidth*devicePixelRatio);canvas.height=Math.round(canvas.clientHeight*devicePixelRatio);
    ctx.fillStyle='#f5f7f6';ctx.fillRect(0,0,canvas.width,canvas.height);
    const corners=[[-2.5,-2.5,0],[2.5,-2.5,0],[2.5,2.5,0],[-2.5,2.5,0]].map(project);
    ctx.fillStyle='#718279';ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();
    for(let u=-2.5;u<=2.5;u+=.5){line([u,-2.5,0],[u,2.5,0],'#bdcbc4',.5);line([-2.5,u,0],[2.5,u,0],'#bdcbc4',.5);}
    for(const [end,color,label] of [[[2,0,0],'#9b3030','X'],[[0,2,0],'#18643e','Y'],[[0,0,2.2],'#34589b','Z']]){
      line([0,0,0],end,color,2);const p=project(end);ctx.fillStyle=color;ctx.font=`${12*devicePixelRatio}px sans-serif`;ctx.fillText(label,p[0]+5,p[1]);}
    for(let u=.5;u<=2;u+=.5)for(const p of [[u,0,0],[0,u,0]]){const q=project(p);ctx.fillStyle='#283c32';ctx.font=`${10*devicePixelRatio}px sans-serif`;ctx.fillText(u.toFixed(1)+'m',q[0]+4,q[1]-4);}
    if($('walker').checked){const n=d.walkerNames.length;for(const [a,b] of d.walkerEdges){const A=point(walker,a,fi,n),B=point(walker,b,fi,n);line(A,B,'#173a42',8);line(A,B,'#65909a',4);}
      const left=point(cameras,0,fi,2),right=point(cameras,1,fi,2);line(left,right,'#444',2);dot(left,'#7c3f93',5);dot(right,'#7c3f93',5);}
    if($('mesh').checked){const tris=[];for(const face of d.faces){const P=face.map(j=>point(vertices,j,fi,6890));if(P.every(valid)){const q=P.map(project);tris.push([q.reduce((s,x)=>s+x[2],0),q]);}}
      tris.sort((a,b)=>b[0]-a[0]);ctx.fillStyle='#c08b65';ctx.globalAlpha=.16;
      for(const [,q] of tris){ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);ctx.lineTo(q[1][0],q[1][1]);ctx.lineTo(q[2][0],q[2][1]);ctx.closePath();ctx.fill();}ctx.globalAlpha=1;}
    let rawEdges=0;if($('raw').checked){for(const [a,b] of E){const A=point(raw,a),B=point(raw,b);if(valid(A)&&valid(B)){line(A,B,'#141414',1);rawEdges++;}}
      for(let j=0;j<17;j++){const p=point(raw,j);if(valid(p))dot(p,d.accepted[fi][j]?'#222':'#b47e24',1.8);}}
    if($('reference').checked&&selected!==2)skeleton(d.models[2],true);
    const edges=skeleton(d.models[selected]);
    const audit={frame:fi,variant:d.models[selected].name,frames:448,modelEdges:edges,rawEdges,displayMap:'X,-Y,Z',sceneFresh:true,
      meshVertices:6890,meshFaces:d.faces.length,meshIndexEntries:d.faces.length*3,meshVisible:$('mesh').checked,
      ankleHeightMm:[15,16].map(j=>point(d.models[selected].xyz,j)[2]*1000),hipMedianPx:median(d.models[selected].errors[fi].flatMap(view=>view.slice(6,8))),
      ankleStatus:d.models[selected].status[fi].slice(10,12),interpolation:false,view:{yaw,pitch,zoom,panX,panY}};
    window.viewerAudit=audit;
    canvas.dataset.audit=JSON.stringify(audit);
  }
  function chart(id,from,to,getValue,label){
    const c=$(id),g=c.getContext('2d'),s=devicePixelRatio;c.width=c.clientWidth*s;c.height=c.clientHeight*s;
    const W=c.width,H=c.height,L=48*s,T=20*s,B=H-30*s,R=W-14*s;
    const values=d.models.flatMap((m,k)=>Array.from({length:to-from+1},(_,j)=>getValue(k,from+j))).filter(x=>x!==null&&Number.isFinite(x));
    let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max(2,(hi-lo)*.08);lo-=pad;hi+=pad;
    const x=i=>L+(R-L)*(d.time[i]-d.time[from])/(d.time[to]-d.time[from]),y=v=>B-(B-T)*(v-lo)/(hi-lo);
    g.fillStyle='#fff';g.fillRect(0,0,W,H);g.font=`${10*s}px sans-serif`;
    for(let k=0;k<=4;k++){const val=lo+(hi-lo)*k/4,yy=y(val);g.strokeStyle='#ddd';g.lineWidth=s;g.beginPath();g.moveTo(L,yy);g.lineTo(R,yy);g.stroke();g.fillStyle='#222';g.fillText(val.toFixed(0),3*s,yy+3*s);}
    g.fillText(label,L,11*s);for(let k=0;k<=4;k++){const i=Math.round(from+(to-from)*k/4);g.fillText(d.time[i].toFixed(1)+'s',x(i)-10*s,H-10*s);}
    for(let m=0;m<3;m++){g.strokeStyle=d.models[m].color;g.lineWidth=(m===selected?2:1.2)*s;g.beginPath();let prev=false;
      for(let i=from;i<=to;i++){const v=getValue(m,i);const broken=i>from&&d.models[m].segments[i].some((value,j)=>value!==d.models[m].segments[i-1][j]);
        if(v===null||!Number.isFinite(v)){prev=false;continue;}if(prev&&!broken)g.lineTo(x(i),y(v));else g.moveTo(x(i),y(v));prev=true;}g.stroke();}
    if(fi>=from&&fi<=to){g.strokeStyle='#333';g.setLineDash([3*s,3*s]);g.beginPath();g.moveTo(x(fi),T);g.lineTo(x(fi),B);g.stroke();g.setLineDash([]);}
    c.onclick=e=>{const f=(e.offsetX*s-L)/(R-L),target=d.time[from]+Math.max(0,Math.min(1,f))*(d.time[to]-d.time[from]);let closest=from;for(let i=from;i<=to;i++)if(Math.abs(d.time[i]-target)<Math.abs(d.time[closest]-target))closest=i;setFrame(closest);};
  }
  function drawMetrics(){
    const m=d.models[selected],ankles=[15,16].map(j=>point(m.xyz,j)[2]*1000);
    $('frame').value=fi;$('seek').value=fi;$('clock').textContent=`${d.time[fi].toFixed(3)} s / ${d.time[447].toFixed(3)} s`;
    $('stage').textContent=({stage1_walker_static_human_moving:'Stage 1 · 助步器静止',stage2_feet_static_walker_moving:'Stage 2 · 助步器移动',transition:'阶段过渡',warming_up:'准备阶段'}[d.stage[fi]]||d.stage[fi]);
    $('ankles').textContent=ankles.map(x=>number(x)).join(' / ');$('hips').textContent=number(median(m.errors[fi].flatMap(view=>view.slice(6,8))));
    $('jointRows').replaceChildren();const labels=['左髋','右髋','左膝','右膝','左踝','右踝'];
    for(let j=6;j<12;j++){const row=document.createElement('tr');const entries=[labels[j-6],number(point(m.xyz,j+5)[2]*1000),m.errors[fi].map(view=>number(view[j])).join('/'),states[m.status[fi][j]]||m.status[fi][j]];
      for(const entry of entries){const td=document.createElement('td');td.textContent=entry;row.append(td);}$('jointRows').append(row);}
    $('pose').textContent=`地面变换：${d.pose[fi]} · ${d.poseReason[fi]||'无拒绝原因'}`;
    const counts={};for(const state of m.status[fi])counts[state]=(counts[state]||0)+1;
    $('quality').textContent=`${m.label}\n${Object.entries(counts).map(([k,v])=>(states[k]||k)+' '+v).join(' / ')}\n`+
      labels.map((name,k)=>`${name}：${m.reason[fi][k+6]||'—'}`).join('\n');
    chart('heightChart',35,61,(k,i)=>{const z=[15,16].map(j=>point(d.models[k].xyz,j,i)[2]);return z.every(Number.isFinite)?(z[0]+z[1])*500:null;},'踝高度 mm');
    chart('hipChart',0,447,(k,i)=>median(d.models[k].errors[i].flatMap(view=>view.slice(6,8))),'髋回投中位 px');
  }
  function upright(xy,side){return side===0?[xy[1],1919-xy[0]]:[1079-xy[1],xy[0]];}
  let imageCache=new Map();
  function inputImages(){const serial=++drawSerial,current=fi;
    for(let side=0;side<2;side++){
      const c=$(side===0?'leftImage':'rightImage'),g=c.getContext('2d'),url=`${d.imageRoot}/${side===0?'left_ccw90':'right_cw90'}/pair_${String(current).padStart(4,'0')}.png`;
      g.fillStyle='#fff';g.fillRect(0,0,c.width,c.height);g.fillStyle='#222';g.fillText('加载当前图像…',10,20);
      const paint=img=>{if(serial!==drawSerial)return;g.drawImage(img,0,0,c.width,c.height);const scale=c.width/1080;
        const xy=points=>points.map(p=>p&&p.slice(0,2).every(x=>x!==null&&Number.isFinite(x))?upright(p,side).map(x=>x*scale):null);
        const observed=xy(d.observed[current][side]),projected=xy(d.models[selected].pixels[current][side]),reference=xy(d.models[2].pixels[current][side]);
        g.strokeStyle='#111';g.lineWidth=1.5;for(const [a,b] of E)if(observed[a]&&observed[b]){g.beginPath();g.moveTo(...observed[a]);g.lineTo(...observed[b]);g.stroke();}
        const dots=(points,color,r)=>{g.fillStyle=color;for(let j=5;j<17;j++)if(points[j]){g.beginPath();g.arc(points[j][0],points[j][1],r,0,Math.PI*2);g.fill();}};
        dots(observed,'#000',2.8);if($('reference').checked&&selected!==2)dots(reference,d.models[2].color,3);dots(projected,d.models[selected].color,4);
      };
      if(imageCache.has(url))paint(imageCache.get(url));else{const image=new Image();image.onload=()=>{imageCache.set(url,image);if(imageCache.size>8)imageCache.delete(imageCache.keys().next().value);paint(image);};image.onerror=()=>{if(serial===drawSerial){g.fillStyle='#b00';g.fillText('当前图像读取失败；未使用上一帧替代',10,40);}};image.src=url;}
    }
  }
  function draw(){drawScene();drawMetrics();inputImages();}
  function setFrame(frame,stop=true){fi=Math.max(0,Math.min(447,Math.round(frame)));if(stop){playing=false;$('play').textContent='播放';}draw();}
  $('frame').oninput=e=>{if(e.target.value!==''&&Number.isFinite(e.target.valueAsNumber))setFrame(e.target.valueAsNumber);};$('seek').oninput=e=>setFrame(+e.target.value);
  document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>setFrame(+b.dataset.frame));
  $('variant').onchange=e=>{selected=+e.target.value;draw();};
  for(const id of ['reference','mesh','raw','walker'])$(id).onchange=draw;
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{const v=b.dataset.view;[yaw,pitch]=({recommended:[.65,.12],side:[Math.PI/2,0],front:[0,0],top:[0,Math.PI/2]}[v]);zoom=.85;panX=panY=0;drawScene();});
  $('play').onclick=()=>{playing=!playing;if(playing&&$('loopKey').checked&&(fi<35||fi>61))fi=35;lastTime=0;$('play').textContent=playing?'暂停':'播放';};
  let drag=null;canvas.oncontextmenu=e=>e.preventDefault();canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
  canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.pan){panX+=dx*devicePixelRatio;panY+=dy*devicePixelRatio;}else{yaw-=dx*.007;pitch+=dy*.007;}drawScene();};
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.15,Math.min(6,zoom*(e.deltaY<0?1.08:.93)));drawScene();},{passive:false});
  function tick(time){if(playing&&time-lastTime>=1000/(30*Number($('rate').value))){const key=$('loopKey').checked;fi=key?(fi>=61||fi<35?35:fi+1):(fi+1)%448;lastTime=time;draw();}requestAnimationFrame(tick);}
  $('loading').hidden=true;$('app').hidden=false;draw();requestAnimationFrame(tick);window.addEventListener('resize',draw);
  new ResizeObserver(()=>{document.documentElement.style.setProperty('--header-height',document.querySelector('header').getBoundingClientRect().height+'px');drawScene();}).observe(document.querySelector('header'));
})().catch(error=>{document.getElementById('loading').textContent='可视化加载失败：'+error.message;window.viewerError=String(error);});
