#!/usr/bin/env python
"""
Test the YAML object instantiation fix
"""

import tempfile
import sys
import os

# Add UAVarPrior to path
sys.path.insert(0, '/home/sdodl001/UAVarPrior/src')

from uavarprior.setup.config import load_path

def test_yaml_instantiation():
    """Test if YAML object instantiation is working"""
    
    # Test YAML file with object instantiation
    yaml_content = '''test_obj: !obj:uavarprior.utils.load_features_list
  input_path: /scratch/ml-csm/projects/fgenom/gve/data/features/kmeans/features_kmeans1.txt
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        
        try:
            print('Loading YAML with instantiate=False...')
            configs_no_inst = load_path(f.name, instantiate=False)
            print(f'test_obj type: {type(configs_no_inst["test_obj"])}')
            if hasattr(configs_no_inst['test_obj'], 'bind'):
                print('✅ test_obj has bind method - it is a _Proxy object!')
                return True
            else:
                print('❌ test_obj does NOT have bind method')
                print(f'test_obj content: {configs_no_inst["test_obj"]}')
                return False
                
        except Exception as e:
            print(f'❌ Error loading YAML: {e}')
            import traceback
            traceback.print_exc()
            return False
        finally:
            os.unlink(f.name)

if __name__ == "__main__":
    test_yaml_instantiation()
