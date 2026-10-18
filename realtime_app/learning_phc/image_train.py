"""CPU PhysX teacher distillation -> direct-image PPO -> frozen full448 replay.

Import the native backend before torch (Isaac Gym requirement). Train data is
physically separated from inference inputs. No teacher fallback in deployment.
"""

from .native import PhysXBody

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .image_policy import ImageObservation, ImageActorCritic
from .policy import FrozenPHC, observation_v7, recovery_reference

ROOT = Path(__file__).resolve().parents[2]


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class TrainingEnvironment:
    def __init__(self, packet, labels, xml, rng):
        if len(labels['reference_position_m']) != 256:
            raise ValueError('only prefix0..255 privileged training references allowed')
        self.packet, self.labels, self.xml, self.rng = packet, labels, xml, rng
        self.observation = ImageObservation(packet)
        self.body = None
        self.resets = 0
        self.wrist_camera = ((packet['initial_position_m'][[17, 22]]-packet['camera_translation_m'][0])
                             @packet['camera_rotation'][0])

    def reset(self):
        if self.body is not None:
            self.body.close()
        start = int(self.rng.integers(0, 192))
        self.body = PhysXBody(self.xml, self.labels['position_m'][start, 0],
                             self.labels['rotation_world'][start, 0])
        self.body.reset_joints(self.labels['rotation_world'][start])
        self.index, self.length = start+1, 0
        self.resets += 1
        return self.observation.features(self.index, *self.body.state())

    def teacher_action(self, teacher):
        if not 1 <= self.index <= 255:
            raise ValueError('heldout teacher query forbidden')
        p, r, v, w = self.body.state()
        target, target_velocity = recovery_reference(p, v,
            self.labels['reference_position_m'][self.index],
            self.labels['reference_velocity_m_s'][self.index])
        return teacher.act(observation_v7(p, r, v, w, target, target_velocity))[0]

    def step(self, action):
        index = self.index
        wrist = self.wrist_camera@self.packet['camera_rotation'][index].T+self.packet['camera_translation_m'][index]
        self.body.advance(action, self.packet['timestamp_s'][index]-self.packet['timestamp_s'][index-1], wrist)
        state = self.body.state()
        if not all(np.isfinite(x).all() for x in state):
            raise RuntimeError('nonfinite training physics; retain failure instead of silently resetting')
        reward, image, fallen = self.observation.reward(index, state[0], state[1], action)
        self.index += 1
        self.length += 1
        done = fallen or self.index >= 256 or self.length >= 64
        # A training episode explicitly terminates at its reset boundary. GAE
        # must not bootstrap through a randomly initialized unrelated state.
        next_observation = np.zeros(574, np.float32) if done else self.observation.features(self.index, *state)
        return next_observation, reward, done, image, fallen

    def close(self):
        if self.body is not None:
            self.body.close()
            self.body = None


def fit_actions(model, observations, targets, steps, rng, log, phase):
    x = torch.tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    first, last = None, None
    for step in range(steps):
        indices = rng.integers(0, len(x), 128)
        loss = torch.mean((model.action(x[indices])-y[indices])**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        optimizer.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    row = dict(phase=phase, samples=len(x), optimizer_steps=steps, first_batch_mse=first, last_batch_mse=last)
    log.write(json.dumps(row)+'\n'); log.flush()
    print(row, flush=True)


def checkpoint(path, model, phase, parameters):
    torch.save(dict(model=model.state_dict(), phase=phase, architecture='574-256-256-69 actor, separate critic',
                    parameters=parameters, inference='image rays and self state; teacher absent'), path)


def train(args, packet):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    parameters = dict(seed=0, bc_steps=600, dagger_rounds=2, dagger_steps=2048,
        dagger_fit_steps=200, ppo_updates=64, ppo_horizon=128, ppo_epochs=4,
        gamma=.97, gae_lambda=.95, clip=.2, ppo_lr=3e-5, teacher_action_loss=.1,
        reward='confidence-weighted exp(-(raw_pixel_error/50)^2)+.1-fallen-.01mean(action^2)',
        teacher_frames=[0,255], inference_packet=str(args.packet.resolve()),
        teacher_labels=str(args.teacher_data.resolve()), xml=str(args.xml.resolve()),
        policy_input='current rays, pixel residuals and self-state; no reference3D, future images or frame index')
    (args.output/'parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    labels = read_arrays(args.teacher_data)
    teacher = FrozenPHC(ROOT/'models/phc/phc_kp_2.pth', ROOT/'models/phc/phc_comp_kp_2.pth')
    model = ImageActorCritic(teacher.primitive_mean, teacher.primitive_var)
    observation = ImageObservation(packet)
    x, y = [], []
    for i in range(1,256):
        x.append(observation.features(i, labels['position_m'][i-1], labels['rotation_world'][i-1],
            labels['velocity_m_s'][i-1], labels['angular_velocity_rad_s'][i-1]))
        y.append(labels['action'][i])
    environment = TrainingEnvironment(packet, labels, args.xml, rng)
    try:
        with (args.output/'training.jsonl').open('w',encoding='utf-8') as log:
            fit_actions(model,x,y,600,rng,log,'initial_teacher_BC')
            for round_id in range(2):
                obs = environment.reset()
                falls = 0
                for step in range(2048):
                    target = environment.teacher_action(teacher)
                    x.append(obs.copy()); y.append(target.copy())
                    with torch.no_grad():
                        action = model.action(torch.tensor(obs)).numpy()
                    # DAgger teacher/student mixing is training only. Final
                    # checkpoints run only the student, including after falls.
                    if rng.random() < (.5 if round_id==0 else .25):
                        action = target
                    obs,_,done,_,fallen = environment.step(action)
                    falls += int(fallen)
                    if done:
                        obs = environment.reset()
                    if (step+1)%512==0:
                        print('DAgger',round_id+1,'samples',step+1,'falls',falls,flush=True)
                fit_actions(model,x,y,200,rng,log,'DAgger'+str(round_id+1))
            checkpoint(args.output/'bc.pt',model,'BC_DAgger',parameters)
            optimizer = torch.optim.Adam(model.parameters(),lr=3e-5)
            obs = environment.reset()
            for update in range(64):
                observations,raws,logps,values,rewards,dones,teachers,images = [],[],[],[],[],[],[],[]
                falls = 0
                for _ in range(128):
                    observations.append(obs.copy())
                    teachers.append(environment.teacher_action(teacher))
                    with torch.no_grad():
                        distribution,value = model.distribution(torch.tensor(obs))
                        raw = distribution.sample()
                        action = raw.tanh().numpy()
                        logp = model.log_probability(distribution,raw)
                    obs,reward,done,image,fallen = environment.step(action)
                    raws.append(raw.numpy()); logps.append(float(logp)); values.append(float(value))
                    rewards.append(reward); dones.append(done); images.append(image); falls+=int(fallen)
                    if done:
                        obs = environment.reset()
                with torch.no_grad():
                    _,bootstrap = model.distribution(torch.tensor(obs))
                advantage = np.zeros(128,dtype=np.float32)
                carry = 0.
                for i in range(127,-1,-1):
                    following = float(bootstrap) if i==127 else values[i+1]
                    continuation = 1.-float(dones[i])
                    delta = rewards[i]+.97*following*continuation-values[i]
                    carry = delta+.97*.95*continuation*carry
                    advantage[i] = carry
                returns = torch.tensor(advantage+np.asarray(values,dtype=np.float32))
                adv = torch.tensor((advantage-advantage.mean())/(advantage.std()+1e-8))
                batch = torch.tensor(np.asarray(observations))
                raw_batch = torch.tensor(np.asarray(raws))
                old_logp = torch.tensor(logps)
                teacher_batch = torch.tensor(np.asarray(teachers))
                kl,actor_loss,value_loss = 0.,None,None
                for _ in range(4):
                    for ids in np.array_split(rng.permutation(128),2):
                        distribution,value = model.distribution(batch[ids])
                        new_logp = model.log_probability(distribution,raw_batch[ids])
                        ratio = torch.exp(new_logp-old_logp[ids])
                        actor_loss = -torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                        value_loss = (value-returns[ids]).square().mean()
                        imitation = (distribution.mean.tanh()-teacher_batch[ids]).square().mean()
                        loss = actor_loss+.5*value_loss+.1*imitation
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                        kl = float((old_logp[ids]-new_logp).mean().detach())
                    if kl>.04:
                        break
                row = dict(phase='PPO',update=update+1,physics_steps=(update+1)*128,
                    train_image_reward=float(np.mean(images)),train_reward=float(np.mean(rewards)),
                    fall_episodes=falls,actor_loss=float(actor_loss.detach()),value_loss=float(value_loss.detach()),
                    approximate_kl=kl,training_resets=environment.resets)
                log.write(json.dumps(row)+'\n');log.flush()
                if (update+1)%4==0:
                    print(row,flush=True)
            checkpoint(args.output/'ppo.pt',model,'BC_DAgger_then_PPO',parameters)
    finally:
        environment.close()
    print('Training complete; checkpoints frozen before any validation/test replay',flush=True)


def replay(packet, xml, model_path, output):
    """No teacher object, training labels or future3D input available here."""
    output.mkdir(exist_ok=False)
    saved = torch.load(model_path,map_location='cpu',weights_only=False)
    model = ImageActorCritic();model.load_state_dict(saved['model']);model.eval()
    observation = ImageObservation(packet)
    body = PhysXBody(xml,packet['initial_position_m'][0],packet['initial_rotation'][0])
    body.reset_joints(packet['initial_rotation'])
    count = len(packet['pair_id'])
    p,r,v,w = np.full((count,24,3),np.nan),np.full((count,24,3,3),np.nan),np.full((count,24,3),np.nan),np.full((count,24,3),np.nan)
    actions,timing = np.full((count,69),np.nan),np.zeros(count)
    available = np.zeros(count,bool)
    statuses = []
    forces = np.zeros(count)
    saturated = np.zeros(count,int)
    wrist_camera = (packet['initial_position_m'][[17,22]]-packet['camera_translation_m'][0])@packet['camera_rotation'][0]
    failed = False
    try:
        with (output/'trajectory.jsonl').open('w',encoding='utf-8') as log:
            for i in range(count):
                start = time.perf_counter()
                if not failed:
                    if i:
                        features = observation.features(i,*body.state())
                        with torch.no_grad():
                            action = model.action(torch.tensor(features)).numpy()
                        wrist = wrist_camera@packet['camera_rotation'][i].T+packet['camera_translation_m'][i]
                        body.advance(action,packet['timestamp_s'][i]-packet['timestamp_s'][i-1],wrist)
                        actions[i] = action
                        forces[i],saturated[i] = body.max_support_force,body.support_saturation_substeps
                    state = body.state()
                    failed = not all(np.isfinite(a).all() for a in state)
                    if not failed:
                        p[i],r[i],v[i],w[i] = state
                        available[i] = True
                status = 'nonfinite_simulation' if failed else ('fallen_finite' if p[i,0,2]<.3 else 'image_policy_step')
                if i==0 and not failed:
                    status='initialized_from_frame0'
                statuses.append(status)
                timing[i]=(time.perf_counter()-start)*1000
                row = dict(pair_id=i,timestamp_s=float(packet['timestamp_s'][i]),status=status,
                    available=bool(available[i]),reference_mean_error_m=None,assumed_wrist_support=True,
                    policy_reinitialized=False,world_pose_certified=False)
                log.write(json.dumps(row)+'\n')
        np.savez_compressed(output/'rollout.npz',pair_id=packet['pair_id'],timestamp_s=packet['timestamp_s'],
            position_m=p,rotation_world=r,velocity_m_s=v,angular_velocity_rad_s=w,action=actions,
            available=available,status=np.asarray(statuses),step_ms=timing)
        np.savez_compressed(output/'support_diagnostic.npz',maximum_force_n=forces,saturated_side_substeps=saturated)
        summary = dict(status='complete448_direct_image_replay;research_candidate',frames=count,
            finite_frames=int(available.sum()),fallen_finite_frames=int(np.sum(p[:,0,2]<.3)),
            weights_frozen=True,weights_phase=saved['phase'],policy_input='current2D/rays/self-state only',
            teacher_runtime=False,future3D_runtime=False,resets_after_frame0=0,assumed_wrist_support=True,
            step_median_ms=float(np.median(timing)),step_p95_ms=float(np.percentile(timing,95)),
            pelvis_min_height_m=float(np.nanmin(p[:,0,2])),support_max_n=float(forces.max()),
            source_checkpoint=str(model_path),xml=str(xml),contact='original collision and assumed wrists; no heel anchors',
            scope='single-video temporal holdout;no independent3D/generalization claim')
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(summary,flush=True)
    finally:
        body.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--teacher-data',type=Path)
    parser.add_argument('--xml',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='replay-only path; does not load teacher labels or weights')
    args = parser.parse_args()
    torch.set_num_threads(2)
    packet = read_arrays(args.packet)
    if args.checkpoint:
        replay(packet,args.xml,args.checkpoint,args.output)
    else:
        if args.teacher_data is None:
            parser.error('train-only teacher data required for training')
        train(args,packet)
        # Both checkpoints are frozen before full sequence is inspected.
        for name in ('bc','ppo'):
            replay(packet,args.xml,args.output/(name+'.pt'),args.output/('replay_'+name))


if __name__ == '__main__':
    main()
