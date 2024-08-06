import os
from pydmclab.hpc.sets import GetSet
from pydmclab.core.struc import StrucTools

class UnitTestGetSet(unittest.TestCase):
    
    def setUp(self) -> None:
        self.test_data_dir = os.path.join(
            HERE, "..", "pydmclab", "data", "test_data", "sets"
        )
        
        self.structure_AlN = StrucTools(
            os.path.join(self.test_data_dir, "Al1N1.vasp")
        ).structure
        
        self.structure_MnO = StrucTools(
            os.path.join(self.test_data_dir, "Mn1O1.vasp")
        ).structure
        
        return
    
    def test_sets(self):


if __name__ == "__main__":
    unittest.main()
