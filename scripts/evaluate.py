import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.evaluator import run_evaluation

if __name__ == '__main__':
    run_evaluation()
