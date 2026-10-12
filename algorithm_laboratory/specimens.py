"""Small, editable specimen collection. Build it once, reuse exact input bytes."""
import argparse
from pathlib import Path
from svdbench.laboratory import MatrixCase
from svdbench.laboratory.trace import write_json


def specimens():
    return {
        'dense': MatrixCase.create(m=64,n=64,family='gaussian'),
        'tall': MatrixCase.create(m=128,n=32,family='gaussian'),
        'wide': MatrixCase.create(m=32,n=64,family='gaussian'),
        'low_rank': MatrixCase.create(m=96,n=64,rank=8,tail=1e-5),
        'rank_deficient': MatrixCase.create(m=40,n=24,rank=4,tail=0),
        'clustered': MatrixCase.create(m=64,n=64,family='clustered',condition=1e8,cluster_size=4,cluster_spread=1e-6),
        'sparse': MatrixCase.create(m=96,n=64,family='sparse',density=.05),
        'zero': MatrixCase.create(m=16,n=12,family='zero'),
        'complex': MatrixCase.create(m=24,n=16,family='gaussian',dtype='complex128'),
        'batch': MatrixCase.create(m=12,n=8,family='gaussian',batch=4),
        'custom_spectrum': MatrixCase.from_spectrum([5,5,1,.1,1e-8,0],m=12,n=8),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    manifest={}
    for name,case in specimens().items():
        case.save(root/name)
        manifest[name]=dict(spec=case.spec,sha256=case.sha256,shape=list(case.array.shape),dtype=str(case.array.dtype))
    write_json(root/'manifest.json',manifest)
    print(f'Saved {len(manifest)} specimens with independent references to {root.resolve()}')


if __name__=='__main__':main()
