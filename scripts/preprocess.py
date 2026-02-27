import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.preprocessing import AegisPreprocessor

if __name__ == '__main__':
    preprocessor = AegisPreprocessor()
    data, meta = preprocessor.process()
