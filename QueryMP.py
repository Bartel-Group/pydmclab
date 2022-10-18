from pymatgen.ext.matproj import MPRester

class MPQuery(object):
    
    def __init__(self, api_key=None):
        
        api_key = api_key if api_key else 'YOUR_API_KEY'
        
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        
def main():
    api_key = '***REMOVED***'
    return MPQuery(api_key).mpr

if __name__ == '__main__':
    mpr = main()

