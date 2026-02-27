import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.export.exporter import main

if __name__ == '__main__':
    main()
