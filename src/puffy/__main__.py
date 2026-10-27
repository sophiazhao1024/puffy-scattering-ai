"""Usage: python -m puffy solve --b 2; python -m puffy benchmark."""
import argparse
import json
from .physics import solve


def main():
    parser=argparse.ArgumentParser(description='Independent finite-ball Yukawa solver and NEW ML benchmark')
    sub=parser.add_subparsers(dest='command',required=True)
    one=sub.add_parser('solve',help='Solve one dimensionless parameter point')
    one.add_argument('--b',type=float,required=True)
    one.add_argument('--kappa',type=float,default=.35)
    one.add_argument('--ychi',type=float,default=1.)
    one.add_argument('--yn',type=float,default=.5)
    bench=sub.add_parser('benchmark',help='Generate solver labels and test a guarded GP surrogate')
    bench.add_argument('--out',default='results/my-run')
    bench.add_argument('--ntrain',type=int,default=24)
    bench.add_argument('--ntest',type=int,default=20)
    bench.add_argument('--active',type=int,default=6)
    bench.add_argument('--seed',type=int,default=2026)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='solve':result=solve(**args)
    else:
        from .experiment import run
        result=run(**args)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
