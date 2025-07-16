#base imports
import matplotlib
import numpy as np
import random
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from ase.optimize import FIRE

#agox imports
from agox import AGOX
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.collectors import ParallelCollector
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.models.descriptors.fingerprint import Fingerprint
from agox.models.GPR import GPR
from agox.models.GPR.kernels import RBF, Noise
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.priors import Repulsive
from agox.postprocessors import ParallelRelaxPostprocess
from agox.samplers import KMeansSampler

from agox.collectors import StandardCollector
from agox.models.descriptors.simple_fingerprint import SimpleFingerprint
from agox.samplers import DistanceComparator, GeneticSampler

from pydmclab.core.struc import StrucTools

#calculator import
from chgnet.model import CHGNetCalculator

matplotlib.use("Agg")


class AgoxTools():

    def __init__(self, 
                 environment,
                 search_alg ="GOFEE",
                 calculator =None,
                 database   =None,
                 model      =None,
                 sampler    =None,
                 generators =None,
                 collector  =None,
                 acquisitor =None,
                 relaxer    =None,
                 evaluator  =None,
                 seed       =None,
                 steps      =None, #for how many steps the evaluator (and/or relaxer) takes
                 population_size: int = 10, #for evolutionary algorithm
                 ):
        """
        Purpose:
            A wrapper for methods using AGOX as a simulator. Includes basic implementation for Random Structure Search (RSS),
            Evolutionary Algorithm (EA) and GOFEE search algorithms. If no observers are input (other than the environment),
            system will default to the provided AGOX demo settings.
        Args:
            environment

            search_alg (str)

            calculator

            database

            model

            sampler

            generators

            collector

            acquisitor

            relaxer

            evaluator

            seed (int)

            steps (int)

            population_size (int)
        """
        
        
        #instantiating required universal args (environment)

        self.environment = environment
        self.search_alg = search_alg

        #instantiating optional universal args (calculator, seed)

        if calculator == None:
            self.calculator = CHGNetCalculator() #defaults to CHGNet
        else: 
            self.calculator = calculator

        if seed == None:
            self.seed = random.randint(1,500)
        else:
            self.seed = seed

        if steps == None:
            self.steps = 5
        else:
            self.steps = steps

        #instantiating the rest of the system based on search algorithm input:

        #for GOFEE (default)
        if self.search_alg == "GOFEE":

            #instantiating database
            if database == None:
                self.database = Database(filename = "db0.db", order = 5)
            else:
                self.database = database

            #instantiating variables that the collector / acquisitor are dependant on (namely, the model, sampler, and generators)
            if((collector == None) or (acquisitor == None)):

                if model == None:
                    descriptor = Fingerprint(environment=self.environment)
                    beta = 0.01
                    k0 = C(beta, (beta, beta)) * RBF()
                    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
                    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))

                    self.model = GPR(
                        descriptor=descriptor,
                        kernel=kernel,
                        database=self.database,
                        prior=Repulsive())

                else:
                    self.model = model


                if sampler == None:
                    sample_size = 10   
                    self.sampler = KMeansSampler(
                        descriptor=descriptor,
                        database=self.database,
                        sample_size=sample_size)
                else:
                    self.sampler = sampler

                if generators == None:

                    rattle_generator = RattleGenerator(**self.environment.get_confinement())
                    random_generator = RandomGenerator(**self.environment.get_confinement(), contiguous=False)

                    self.generators = [random_generator, rattle_generator]
                else:
                    self.generators = generators

            #instantiating the collector
            if collector == None:
                self.collector = ParallelCollector(
                    generators=self.generators,
                    sampler=self.sampler,
                    environment=self.environment,
                    num_candidates={0: [10, 0], 5: [3, 7]},
                    order=1)
            else:
                self.collector = collector

            #instantiating the acquisitor
            if acquisitor == None:
                self.acquisitor = LowerConfidenceBoundAcquisitor(model=self.model, kappa=2, order=3)
            else:
                self.acquisitor = acquisitor

            #instantiating the relaxer
            if relaxer == None:
                self.relaxer = ParallelRelaxPostprocess(
                    model=self.acquisitor.get_acquisition_calculator(),
                    optimizer = FIRE,
                    constraints=self.environment.get_constraints(),
                    optimizer_run_kwargs={"steps": self.steps},
                    start_relax=8,
                    order=2)
            else:
                self.relaxer = relaxer

            #instantiating the evaluator
            if evaluator == None:
                self.evaluator = LocalOptimizationEvaluator(
                    self.calculator,
                    gets={"get_key": "prioritized_candidates"},
                    optimizer_kwargs={"logfile": None},
                    optimizer_run_kwargs={"fmax": 0.05, "steps": self.steps},
                    constraints=self.environment.get_constraints(),
                    store_trajectory=True,
                    order=4)
            else:
                self.evaluator = evaluator

        #for evolutionary algorithm (EA)
        elif self.search_alg == "EA":

            #instantiating database
            if database == None:
                self.database = Database(filename = "db0.db", order = 3)
            else:
                self.database = database


            #instantiating sampler
            if sampler == None:
                descriptor = SimpleFingerprint(environment=self.environment)
                comparator = DistanceComparator(descriptor, threshold=0.5)

                self.sampler = GeneticSampler(
                    population_size=population_size,
                    comparator=comparator,
                    order=4,
                    database=self.database
                )
            else:
                self.sampler = sampler
            
            #instantiating generators
            if generators == None:

                rattle_generator = RattleGenerator(**self.environment.get_confinement())
                random_generator = RandomGenerator(**self.environment.get_confinement())
                self.generators = [random_generator, rattle_generator]

            else:
                self.generators = generators

            #instantiating collector
            if collector == None:
                num_candidates = {
                    0: [population_size, 0],
                    5: [2, population_size - 2],
                    10: [0, population_size]
                    }

                self.collector = StandardCollector(
                    generators=generators,
                    sampler=sampler,
                    environment=environment,
                    num_candidates=num_candidates,
                    order=1
                    )
            else:
                self.collector = collector

            #instantiating evaluator
            if evaluator == None:
                evaluator = LocalOptimizationEvaluator(
                    self.calculator,
                    gets={"get_key": "candidates"},
                    optimizer_run_kwargs={"fmax": 0.05, "steps": self.steps},
                    store_trajectory=False,
                    order=2,
                    constraints=self.environment.get_constraints(),
                    number_to_evaluate=population_size,
                    )
            else:
                self.evaluator = evaluator

        #for random structure search
        elif self.search_alg == "RSS":

            #instantiating database
            if database == None:
                self.database = Database(filename = "db0.db", order = 3)
            else:
                self.database = database

            #instantiating random generator
            if generators == None:
                self.generators = RandomGenerator(
                    **self.environment.get_confinement(),
                    environment=self.environment,
                    order=1,
                    contiguous=False)
            else:
                self.generators = generators
            
            #instantiating evaluator
            if evaluator == None:
                self.evaluator = LocalOptimizationEvaluator(
                    self.calculator,                                
                    gets={"get_key": "candidates"},
                    optimizer_run_kwargs={"fmax": 0.05, "steps": self.steps},
                    store_trajectory=False,
                    order=2,
                    constraints=self.environment.get_constraints()
                    )
            else:
                self.evaluator = evaluator
        
        #catch-all if a new search algorithm was input (not sure if we should handle this differently?)
        else:
            raise ValueError("Unknown (or not implemented) search algorithm.")
        

        #setting up AGOX framework
        if self.search_alg == "GOFEE":
            self.system = AGOX(self.collector, self.relaxer, self.acquisitor, self.evaluator, self.database, seed=self.seed)

        elif self.search_alg == "EA":
            self.system = AGOX(self.collector, self.database, self.evaluator, seed=self.seed)

        elif self.search_alg == "RSS":
            self.system = AGOX(self.generators, self.database, self.evaluator, seed=self.seed)


    def run_simulation(self, N_iterations:int = 10):

        """
        Args:
            N_iterations (int)
                number of iterations the AGOX simulator will run.
        """

        if self.search_alg == "GOFEE":
            self.system = AGOX(self.collector, self.relaxer, self.acquisitor, self.evaluator, self.database, seed=self.seed)

        elif self.search_alg == "EA":
            self.system = AGOX(self.collector, self.database, self.evaluator, seed=self.seed)

        elif self.search_alg == "RSS":
            self.system = AGOX(self.generators, self.database, self.evaluator, seed=self.seed)
    

        self.system.run(N_iterations = N_iterations)

    def get_best_structure(self):

        """
        Method for retrieving the best structure found after a given run. 
        Note this function only works when a run has been executed on the agox object.
        """

        return Structure.from_ase_atoms(self.database.get_best_structure())

    def database_to_cif(
            self,
            output_filename = None,
            full_trajectory: bool = False):
        """
        Convert a database to a .cif format
        Args:
            output_filename (str)
                the output name of the .cif file created. Defaults to "converted_{database filename}"
            full_trajectory (bool)
                Whether or not the .cif file includes every iteration, or just the best (i.e. lowest energy) iteration.
        """

        from ase.io import write

        #handling name
        if output_filename == None:
            name = f"converted_{self.database.filename}"
        else:
            name = output_filename
        
        #saving best structure .cif file 
        if full_trajectory == False:

            StrucTools(Structure.from_ase_atoms(self.database.get_best_structure())).structure_to_cif(name)

            print(f"Saving best structure to .cif file {name}")

        else:
            trajectory = self.database.restore_to_trajectory()
            write(name+".cif", trajectory, format='cif')

            print(f"Saving full trajectory to .cif file {name}")
            print(f"Saving {len(trajectory)} total structures")


class AgoxHelper():

    def get_env_from_structure(
        structure,
        symbols: str,
        confinement_cell_height: int = 4):

        """
        Initializes an agox environment object using a given Structure object.
        Args:
            structure (Structure)
                the bulk / slab structure to make the environment out of.
            symbols (str)
                the additional atoms to be placed on top of the structure.
            confinement_cell_height (int)
                the height of the confinement cell. Defaults to 4.
        """

        from ase.constraints import FixAtoms 
            
        template = AseAtomsAdaptor().get_atoms(structure,msonable=False)
        template.pbc = [True, True, False]
        template.positions[:, 2] -= template.positions[:, 2].min()

        confinement_cell = template.cell.copy()
        confinement_cell[2, 2] = confinement_cell_height

        z0 = template.positions[:, 2].max()
        confinement_corner = np.array([0, 0, z0])  # Confinement corner is at cell origin

        z_temp = template.positions[:,2].copy()
        z_temp.sort()

        constraint = FixAtoms(mask=[atom.position[2] < z_temp[-2] for atom in template])
        environment = Environment(
            template=template,
            symbols=symbols,
            confinement_cell=confinement_cell,
            confinement_corner=confinement_corner,
            box_constraint_pbc=[True, True, False],  # Confinement is not periodic in z
            fix_template=False,
            constraints=[constraint])

        environment.plot()
        return environment
    
    def run_and_relax(agox,
                      post_relaxer,
                      run_iterations: int = 10,
                      relaxation_steps: int = 200,
                      relaxation_fmax: float = 0.1,
                      save_int_files: bool = False,
                      ):
        
        agox.run_simulation(run_iterations)

        if save_int_files:
            agox.database_to_cif(output_filename=f"int_{agox.database.filename}_full",full_trajectory=True)
            agox.database_to_cif(output_filename=f"int_{agox.database.filename}_best",full_trajectory=False)
        
        run_one_best = agox.get_best_structure()
        results = post_relaxer.relax(
            run_one_best,
            fmax=relaxation_fmax,
            steps=relaxation_steps,
            verbose=True,
        )

        if save_int_files:
            StrucTools(results["final_structure"]).structure_to_cif(f"{agox.database.filename}_relaxed")

        return results["final_structure"]
        