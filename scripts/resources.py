"""Download public weights or share the trained model and demonstration data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import ROOT, SAM_CHECKPOINT, APPEARANCE_WEIGHTS, CLASSIFIER_MODEL, IMAGE_DIR, DATA_DIR

PUBLIC_WEIGHTS = [(SAM_CHECKPOINT, 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'),
                  (APPEARANCE_WEIGHTS, 'https://download.pytorch.org/models/resnet18-f37072fd.pth')]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def download():
    for target, url in PUBLIC_WEIGHTS:
        if target.exists():
            print('Already present:', target); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.part')
        try:
            print('Downloading:', url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, open(temporary, 'wb') as f:
                for chunk in iter(lambda: response.read(1024 * 1024), b''): f.write(chunk)
            if temporary.stat().st_size < 1024 * 1024: raise ValueError('Downloaded file is unexpectedly small')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print('Saved:', target, 'sha256:', digest(target))

def export_bundle(target):
    paths = {'asset/sam_vit_b_01ec64.pth': SAM_CHECKPOINT,
             'asset/resnet18-f37072fd.pth': APPEARANCE_WEIGHTS,
             'asset/dota_classifier/model.joblib': CLASSIFIER_MODEL}
    for p in (ROOT/'dataset/images').glob('practice_*.png'):
        paths[p.relative_to(ROOT).as_posix()] = p
        ann = ROOT/'results'/f'{p.stem}.json'
        if ann.exists(): paths[ann.relative_to(ROOT).as_posix()] = ann
    video = ROOT/'dataset/videos/VisDrone_0011.mp4'
    if video.exists(): paths[video.relative_to(ROOT).as_posix()] = video
    for name in ('report.md', 'evaluation.json'):
        p = CLASSIFIER_MODEL.parent/name
        if p.exists(): paths['asset/dota_classifier/'+name] = p
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError('Missing resources: '+', '.join(missing))
    target = Path(target).resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise FileExistsError(f'Bundle already exists: {target}')
    manifest = {name: {'sha256': digest(p), 'size': p.stat().st_size} for name,p in paths.items()}
    temporary = target.with_suffix('.part')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name,p in paths.items(): z.write(p, name)
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(temporary,target)
    finally: temporary.unlink(missing_ok=True)
    print('Bundle:',target, 'bytes:',target.stat().st_size)

def import_bundle(source):
    with zipfile.ZipFile(source) as z:
        manifest = json.loads(z.read('manifest.json'))
        # Validate every file and existing destination before changing the checkout.
        with tempfile.TemporaryDirectory(prefix='lab2-resources-') as td:
            staged = []
            for name, info in manifest.items():
                parts = Path(name).parts
                if '\\' in name or Path(name).is_absolute() or '..' in parts or not parts or parts[0] not in ('asset','dataset','results'):
                    raise ValueError('Invalid bundle path: '+name)
                target = (ROOT/name).resolve()
                if not target.is_relative_to(ROOT): raise ValueError('Destination outside repository')
                temp = Path(td)/name;temp.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as inp, open(temp,'wb') as out:
                    for chunk in iter(lambda: inp.read(1024*1024),b''): out.write(chunk)
                if temp.stat().st_size != info['size'] or digest(temp) != info['sha256']:
                    raise ValueError('Checksum mismatch: '+name)
                if target.exists():
                    if digest(target) != info['sha256']: raise FileExistsError('Refusing to overwrite existing file: '+str(target))
                    continue
                staged.append((temp,target))
            import shutil
            for temp,target in staged:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(temp,target)
    print('Imported resources; existing identical files preserved.')

def check():
    ready=True
    for label,path in [('SAM',SAM_CHECKPOINT),('Appearance weights',APPEARANCE_WEIGHTS),('Classifier',CLASSIFIER_MODEL)]:
        ok=path.is_file();ready &= ok;print(('OK' if ok else 'MISSING'),label,path)
    print('Image directory:',IMAGE_DIR)
    print('DOTA archive directory:',DATA_DIR/'images')
    if not ready: print('Import a group runtime bundle, or download weights and train the classifier.')
    return 0 if ready else 1

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('download');sub.add_parser('check')
    ex=sub.add_parser('export');ex.add_argument('--out',default='results/local_run/lab2-runtime.zip')
    im=sub.add_parser('import');im.add_argument('bundle')
    args=ap.parse_args()
    if args.command=='download':download()
    elif args.command=='export':export_bundle(args.out)
    elif args.command=='import':import_bundle(args.bundle)
    else: return check()
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        print('Resource error:',e,file=sys.stderr);sys.exit(1)
