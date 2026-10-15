function drawMotion() {
  const j = Number(document.getElementById('joint').value);
  const raw = document.getElementById('raw').checked;
  const velocities = raw ? d.motion.rawVelocity : d.motion.velocity;
  const accelerations = raw ? d.motion.rawAcceleration : d.motion.acceleration;
  const reasonLabels = {available:'可发布', derivative_warmup:'重启热身',
    poor_or_stale_pixel_fit:'像素拟合差或已过期', no_recent_own_pixels:'近期无本点像素',
    constraint_conflict:'地面约束冲突', feasible_unconverged:'几何可行但优化未收敛',
    high_position_uncertainty:'位置不确定性过大'};
  document.getElementById('motion-status').textContent =
    `帧 ${fi} · ${d.motion.time[fi].toFixed(3)} s · ${d.motion.names[j]} · ` +
    (reasonLabels[d.motion.reason[fi][j]] || d.motion.reason[fi][j]);
  document.getElementById('time').textContent = `${d.motion.time[fi].toFixed(3)} s`;
  document.getElementById('wrist').textContent =
    ({initialized:'初始化', updated:'二维反馈', predicted:'仅预测', reinitialized:'重新初始化', feasible_unconverged:'几何可行·未收敛', expired:'失效'}[d.motion.status[fi]] || d.motion.status[fi]);
  document.getElementById('e3').textContent = d.reprojection[fi].map(x=>x===null?'—':x.toFixed(1)).join(' / ') + ' px';
  document.getElementById('hr').textContent = d.motion.valid[fi].filter(Boolean).length + ' / 12';
  document.getElementById('fr').textContent = d.motion.soleHeight[fi].map(x=>x===null?'—':x.toFixed(1)).join(' / ') + ' mm';
  document.getElementById('walker-height').textContent = d.motion.walkerFootHeight[fi].map(x=>x===null?'—':x.toFixed(1)).join(' / ') + ' mm';
  const viewLabels={first_frame_initialization_pixels_not_reused:'首帧像素已用于初始化',
    stereo_epipolar_rejected:'双目极线拒绝', nonlinear_update_rejected:'非线性更新拒绝',
    insufficient_joint_support:'关节点支持不足', low_or_invalid_score:'分数不合格',
    out_of_raw_image_bounds:'原像素越界', non_finite_pixels:'像素非有限'};
  const viewCounts={};
  for(const reason of d.motion.viewReason[fi].flat()) {
    if(!reason.startsWith('used_2d')) viewCounts[reason]=(viewCounts[reason]||0)+1;
  }
  document.getElementById('qa').textContent =
    `二维更新视图：${d.motion.used[fi].flat().filter(Boolean).length}/24\n` +
    `严格几何通过：${d.acc.slice(fi*17,fi*17+17).filter(Boolean).length}/17\n` +
    `跟踪观测状态：${Object.entries(viewCounts).map(([reason,count])=>(viewLabels[reason]||reason)+' ×'+count).join('；') || '全部用于二维反馈'}\n` +
    `地面状态：${d.pose[fi]}；${d.ground_reasons[fi] || '无拒绝原因'}\n` +
    `导数不可用：${d.motion.names.map((name,k)=>d.motion.valid[fi][k]?'':name+'：'+(reasonLabels[d.motion.reason[fi][k]]||d.motion.reason[fi][k])).filter(Boolean).join('；') || '无'}`;
  if(d.motion.world) {
    const c=d.motion.groundConstraint[fi];
    const mm=values=>values ? values.map(x=>x===null?'—':(x*1000).toFixed(2)).join(' / ') : '未设锚';
    document.getElementById('qa').textContent += `\n固定地面约束：${c.status}；实际满足：${c.constraint_satisfied?'是':'否'}\n整脚底最大位移 L/R：${mm(c.sole_max_displacement_m)} mm（上限${d.motion.footLimitMm}mm）\n优化收敛：${c.optimizer_success===undefined?'未执行':c.optimizer_success?'是':'否'}；世界轨迹未经独立认证`;
    document.getElementById('motion-status').style.color=c.constraint_satisfied?'#000000':'#990000';
    if(!c.constraint_satisfied) document.getElementById('motion-status').textContent += ' · 保留的失败估计';
    window.viewerAudit.constraintSatisfied=c.constraint_satisfied;
    window.viewerAudit.constraintStatus=c.status;
  }
  plotMotion('velocity-chart', velocities, j, '速度 mm/s');
  plotMotion('acceleration-chart', accelerations, j, '加速度 mm/s²');
  window.viewerAudit.derivativeAvailable = d.motion.valid[fi].filter(Boolean).length;
  window.viewerAudit.segment = d.motion.segment[fi];
  window.viewerAudit.timestamp = d.motion.time[fi];
}

function plotMotion(id, values, joint, label) {
  const chart = document.getElementById(id), g = chart.getContext('2d');
  chart.width = Math.round(chart.clientWidth * devicePixelRatio);
  chart.height = Math.round(chart.clientHeight * devicePixelRatio);
  const W = chart.width, H = chart.height, scale = devicePixelRatio;
  const left = 68*scale, right = W-18*scale, top = 27*scale, bottom = H-34*scale;
  const all = values.flatMap(frame=>frame[joint].filter(x=>x!==null && Number.isFinite(x)));
  const extent = Math.max(1, ...all.map(Math.abs));
  const x = time=>left+(right-left)*time/d.motion.time[N-1];
  const y = value=>(top+bottom)/2-value*(bottom-top)/(2*extent*1.08);
  g.fillStyle = '#ffffff'; g.fillRect(0,0,W,H); g.font = `${11*scale}px sans-serif`;
  g.lineWidth = scale; g.strokeStyle = '#000000'; g.fillStyle = '#000000';
  g.beginPath(); g.moveTo(left,top); g.lineTo(left,bottom); g.lineTo(right,bottom); g.stroke();
  g.fillText(label, left, 16*scale);
  for(let k=-2;k<=2;k++) {
    const val=extent*k/2, yy=y(val);
    g.fillText(val.toFixed(Math.abs(extent)>100?0:1), 3*scale, yy+4*scale);
    g.strokeStyle='#dddddd'; g.beginPath(); g.moveTo(left,yy); g.lineTo(right,yy); g.stroke();
  }
  for(let k=0;k<=4;k++) {
    const tt=d.motion.time[N-1]*k/4;
    g.fillText(tt.toFixed(1)+'s',x(tt)-8*scale,H-10*scale);
  }
  const colors=['#000000','#555555','#999999'], patterns=[[],[5*scale,3*scale],[2*scale,3*scale]];
  for(let axis=0;axis<3;axis++) {
    g.strokeStyle=colors[axis]; g.lineWidth=1.4*scale; g.setLineDash(patterns[axis]);
    g.beginPath(); let connected=false;
    for(let k=0;k<N;k++) {
      const value=values[k][joint][axis];
      if(value===null || !Number.isFinite(value)) {connected=false; continue;}
      if(k && d.motion.segment[k]!==d.motion.segment[k-1]) connected=false;
      const xx=x(d.motion.time[k]),yy=y(value);
      if(connected) g.lineTo(xx,yy); else g.moveTo(xx,yy);
      connected=true;
    }
    g.stroke();
    const value=values[fi][joint][axis];
    if(value!==null && Number.isFinite(value)) {
      g.fillStyle=colors[axis]; g.beginPath(); g.arc(x(d.motion.time[fi]),y(value),3*scale,0,2*Math.PI); g.fill();
    }
  }
  g.setLineDash([]); g.strokeStyle='#000000'; g.lineWidth=scale;
  g.beginPath(); g.moveTo(x(d.motion.time[fi]),top); g.lineTo(x(d.motion.time[fi]),bottom); g.stroke();
}

let imageFrame = -1;
function renderInputs() {
  if(imageFrame===fi) return;
  imageFrame=fi;
  const frame=fi;
  ['left','right'].forEach((side,c)=>{
    const target=document.getElementById(side+'-image'),g=target.getContext('2d');
    target.width=270; target.height=480;
    g.fillStyle='#ffffff';g.fillRect(0,0,270,480);g.fillStyle='#000000';
    g.fillText(`加载帧 ${frame}…`,10,30);
    const image=new Image();
    image.onload=()=>{
      if(fi!==frame) return;
      g.drawImage(image,0,0,270,480);
      const obs=d.motion.observed[frame][c],model=d.motion.projected[frame][c];
      const up=point=>c===0?[point[1]/4,(1919-point[0])/4]:[(1079-point[1])/4,point[0]/4];
      g.strokeStyle='#000000';g.lineWidth=1;
      for(const [a,b] of E) {
        if(!obs[a].slice(0,2).every(Number.isFinite)||!obs[b].slice(0,2).every(Number.isFinite)) continue;
        const A=up(obs[a]),B=up(obs[b]);g.beginPath();g.moveTo(...A);g.lineTo(...B);g.stroke();
      }
      for(let k=0;k<17;k++) {
        if(obs[k].slice(0,2).every(Number.isFinite)) {
          const P=up(obs[k]);g.fillStyle='#ffffff';g.beginPath();g.arc(...P,2.5,0,2*Math.PI);g.fill();g.strokeStyle='#000000';g.stroke();
        }
        if(model[k].every(x=>x!==null && Number.isFinite(x))) {
          const P=up(model[k]);g.strokeStyle='#0077aa';g.beginPath();g.moveTo(P[0]-3,P[1]-3);g.lineTo(P[0]+3,P[1]+3);g.moveTo(P[0]-3,P[1]+3);g.lineTo(P[0]+3,P[1]-3);g.stroke();
        }
      }
      window.viewerAudit[side+'ImageFrame']=frame;
    };
    image.onerror=()=>{if(fi===frame){g.fillStyle='#000000';g.fillText(`图像不可用：帧 ${frame}`,10,50);}};
    image.src=d.motion.imageRoot+'/'+(c===0?'left_ccw90':'right_cw90')+'/pair_'+String(frame).padStart(4,'0')+'.png';
  });
}

document.getElementById('joint').onchange=draw;
document.getElementById('raw').onchange=draw;
document.getElementById('frame-number').oninput=e=>{
  fi=Math.max(0,Math.min(N-1,Math.round(Number(e.target.value))));
  playing=false;document.getElementById('play').textContent='播放';
  document.getElementById('slider').value=fi;draw();
};
document.querySelectorAll('[data-motion-view]').forEach(button=>button.onclick=()=>{
  const angles={recommended:[.4,.4],front:[0,0],side:[Math.PI/2,0],top:[0,Math.PI/2]};
  [yaw,pitch]=angles[button.dataset.motionView];panX=panY=0;draw();
});
