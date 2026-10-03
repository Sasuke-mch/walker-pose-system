"""Small, exact-position wrist markers on the sparse annotation source images."""
import argparse
import html
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose_app.body_observations import raw_side
from pose_app.smplh_hand_observation import read_view
from walker_tools.hands.solve_multiframe_wrist_targets import raw_to_upright


COLORS = {'人工': (255, 55, 55), 'PMPose': (255, 205, 0),
          'Sapiens2': (0, 210, 80), 'WiLoR': (0, 180, 255)}


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def mark(image, points, offset=(0, 0), scale=1):
    array = np.array(image.convert('RGB')).copy()
    for name, p in points.items():
        if p is None:
            continue
        xy = (np.asarray(p)-offset)*scale
        if np.isfinite(xy).all():
            cv2.circle(array, tuple(np.round(xy*16).astype(int)), 2*16,
                       COLORS[name], -1, cv2.LINE_AA, shift=4)
    return Image.fromarray(array)


def footer(image, text):
    out = Image.new('RGB', (image.width, image.height+54), 'white')
    out.paste(image, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((12, image.height+4), text, fill='black', font=font(14))
    for i, (name, color) in enumerate(COLORS.items()):
        x = 18+i*112
        d.ellipse((x-2, image.height+36-2, x+2, image.height+36+2), fill=color)
        d.text((x+9, image.height+26), name, fill='black', font=font(14))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('refuse existing output')
    protocol = json.loads((args.comparison/'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((Path(protocol['annotations'])/'manifest.json').read_text(encoding='utf-8'))
    comparison = json.loads((args.comparison/'points.json').read_text(encoding='utf-8'))
    rows = {(r['frame_index'], r['camera'], r['hand']): r for r in comparison}
    points_by_camera = {}
    for camera in ('left', 'right'):
        body = {}
        for method in ('PMPose', 'Sapiens2'):
            side = raw_side(Path(protocol['inputs'][method][camera+'_raw']), camera)
            n = protocol.get('sequence_frames', 448)
            if sorted(side) != list(range(n)):
                raise ValueError('source frame range differs')
            body[method] = np.stack([side[f] for f in range(n)])
        hand, _, _ = read_view(Path(protocol['inputs']['PMPose']['wilor_'+camera]), body['PMPose'], (1920, 1080), camera)
        points_by_camera[camera] = {'PMPose': body['PMPose'][:, 9:11, :2],
                                  'Sapiens2': body['Sapiens2'][:, 9:11, :2], 'WiLoR': hand[:, :, 0]}
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    (args.output/'crops').mkdir()
    entries, checks, preview = [], [], []
    for e in manifest['images']:
        f, camera = e['frame_index'], e['camera']
        source = Image.open(e['source']).convert('RGB')
        if source.size != (1080, 1920):
            raise ValueError('source image dimensions differ')
        all_points, crop_entries, states = [], [], []
        for h, side in enumerate(('left', 'right')):
            row = rows[f, camera, side]
            points = {'人工': row['manual'].get('upright_px')}
            for method in ('PMPose', 'Sapiens2', 'WiLoR'):
                p = points_by_camera[camera][method][f, h]
                points[method] = raw_to_upright(p, camera).tolist() if np.isfinite(p).all() else None
                expected = row['methods'][method]
                if expected['state'] == 'compared':
                    if not np.allclose(p, expected['prediction_raw_px'], atol=1e-6, rtol=0):
                        raise ValueError('visual point differs from evaluated point')
                    checks.append((f, camera, side, method))
            all_points.append(points)
            unavailable = [name for name, p in points.items() if p is None]
            states.append({'frame': f, 'camera': camera, 'hand': side, 'upright_points': points, 'missing': unavailable})
            if points['人工'] is not None:
                present = np.array([p for p in points.values() if p is not None])
                lo, hi = present.min(0)-60, present.max(0)+60
                center = (lo+hi)/2
                size = np.maximum(hi-lo, [260, 190])
                lo = np.maximum(np.floor(center-size/2), 0).astype(int)
                hi = np.minimum(np.ceil(center+size/2), source.size).astype(int)
                crop = source.crop(tuple(lo)+tuple(hi))
                crop = crop.resize((crop.width*2, crop.height*2), Image.Resampling.LANCZOS)
                crop = mark(crop, points, offset=lo, scale=2)
                label = f'{f:04d} · {"左" if camera=="left" else "右"}相机 · {"左" if side=="left" else "右"}手'
                if unavailable:
                    label += ' · 缺少：'+','.join(unavailable)
                crop = footer(crop, label)
                name = f'{f:04d}_{camera}_{side}.png'
                crop.save(args.output/'crops'/name)
                crop_entries.append({'file': 'crops/'+name, 'label': label})
                if f == 0:
                    preview.append(crop)
        image = source
        for points in all_points:
            image = mark(image, points)
        image = footer(image, f'{f:04d} · {"左" if camera=="left" else "右"}相机')
        name = f'{f:04d}_{camera}.png'
        image.save(args.output/'images'/name)
        entries.append({'frame': f, 'camera': camera, 'file': 'images/'+name, 'crops': crop_entries, 'points': states})
    expected_checks = sum(r['methods'][m]['state'] == 'compared'
                          for r in comparison for m in ('PMPose', 'Sapiens2', 'WiLoR'))
    if len(checks) != expected_checks or len(entries) != len(manifest['images']):
        raise ValueError('visualized evaluated point count differs')
    width = max(p.width for p in preview)
    height = max(p.height for p in preview)
    sheet = Image.new('RGB', (width*2, height*2), 'white')
    for i, p in enumerate(preview):
        sheet.paste(p, ((i%2)*width, (i//2)*height))
    sheet.save(args.output/'preview.png')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>腕点对照</title>
<style>body{font:15px system-ui;margin:24px;background:#fff;color:#111}h1{font-size:22px}nav{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ddd}a{color:#222;margin-right:15px}section{margin:32px 0}summary{cursor:pointer}.crops{display:flex;flex-wrap:wrap;gap:16px}.crops img{max-width:560px;width:100%;height:auto}figure{margin:12px 0}figcaption{margin:6px 0;font-size:13px}.full{max-width:380px;width:100%;height:auto}.dot{display:inline-block;width:5px;height:5px;border-radius:50%;margin:0 6px 2px 16px}</style>
<h1>腕点对照</h1><p>红：人工　黄：PMPose　绿：Sapiens2　蓝：WiLoR。点半径2像素，未移动点位；局部图放大2倍。没有结果的位置不补点。</p><nav>'''
    document += ''.join(f'<a href="#f{f}">{f}</a>' for f in manifest['frames'])+'</nav>'
    for f in manifest['frames']:
        document += f'<section id="f{f}"><h2 style="font-size:18px">{f:04d}</h2><div class="crops">'
        for e in entries:
            if e['frame'] != f:
                continue
            for c in e['crops']:
                document += f'<figure><a href="{c["file"]}" target="_blank"><img src="{c["file"]}"></a><figcaption>{html.escape(c["label"])}</figcaption></figure>'
        document += '</div><details><summary>完整原图</summary>'
        for e in entries:
            if e['frame'] == f:
                document += f'<a href="{e["file"]}" target="_blank"><img class="full" src="{e["file"]}"></a>'
        document += '</details></section>'
    (args.output/'index.html').write_text(document+'</html>', encoding='utf-8')
    audit = {'comparison': str(args.comparison), 'source_annotations': protocol['annotations'],
             'original_images': len(entries), 'manual_crops': sum(len(e['crops']) for e in entries),
             'evaluated_prediction_coordinates_verified': len(checks), 'marker_radius_display_px': 2,
             'points_moved': False, 'missing_points_filled': False, 'entries': entries}
    (args.output/'render_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with zipfile.ZipFile(args.output/'腕点.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path.suffix in ('.png', '.html', '.json'):
                archive.write(path, path.relative_to(args.output))
    print(json.dumps({k: audit[k] for k in ('original_images', 'manual_crops', 'evaluated_prediction_coordinates_verified')}))


if __name__ == '__main__':
    main()
