import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.generator import AegisDataGenerator
from core.config import PROJECT_ROOT

if __name__ == '__main__':
    gen = AegisDataGenerator(n_payers=15000, n_merchants=1000)
    df = gen.generate()
    gen.save(df, PROJECT_ROOT)
