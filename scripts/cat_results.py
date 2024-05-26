import sys
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
import json

RESULT_DIR = Path('/mnt/data/eth3d_outputs/ablations/neuralangelo_eth3d')

# for result_file in sorted(RESULT_DIR.iterdir()):
#     if not result_file.name.endswith('.json'):
#         continue
#     with open(result_file, 'r') as f:
#         results = json.load(f)
#     print(result_file)
#     reports = ['psnr', 'ssim', 'lpips']
#     tol_002_reports = ['completeness', 'accuracy', 'f1']
#     for report in reports:
#         print(results[report])
#     if 'tol_0.02' in results:
#         for report in tol_002_reports:
#             print(results['tol_0.02'][report])


for scene in sorted(RESULT_DIR.iterdir()):
    result_file = scene / 'output' / 'results' / 'results.json'
    if not result_file.name.endswith('.json'):
        continue
    with open(result_file, 'r') as f:
        results = json.load(f)
    print(result_file)
    reports = ['test/psnr', 'test/ssim', 'test/lpips']
    tol_002_reports = ['completeness', 'accuracy', 'f1']
    for report in reports:
        print(results[report])
    if 'tol_0.02' in results:
        for report in tol_002_reports:
            print(results['tol_0.02'][report])
