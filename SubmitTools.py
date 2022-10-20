
class SubmitTools(object):
    
    def __init__(self,
                 launch_dir,
                 machine='msi',
                 job_name='my-job',
                 partition='small',
                 nodes=1,
                 tasks_per_node=24,
                 walltime='24:00:00',
                 mem_per_core=None,
                 f_error='log.e',
                 f_out='log.o',
                 constraint=None,
                 qos=None,
                 account='cbartel'):
        
        self.launch_dir = launch_dir
        self.machine = machine
        self.msi_machines = ['msi', 'mesabi', 'mangi', 'agate']
        self.accounts = {m : 'cbartel' for m in self.msi_machines}
        self.base_options = {'job_name' : job_name,
                            'partition' : partition,
                            'nodes' : nodes,
                            'tasks_per_node' : tasks_per_node,
                            'walltime' : walltime,
                            'mem_per_core' : mem_per_core,
                            'f_error' : f_error,
                            'f_out' : f_out,
                            'constraint' : constraint,
                            'qos' : qos,
                            'account' : account,    
                            'total_tasks' : nodes * tasks_per_node}

    @property
    def manager(self):
        if self.machine in self.msi_machines:
            return '#SBATCH'
        else:
            raise NotImplementedError('Machine not supported')
    
    @property
    def partitions(self):
        
        partitions = {}
        partitions['agate'] = {}
        partitions['agate']['agsmall'] = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 96,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 1}
        
        partitions['agate']['aglarge']  = {'cores_per_node' : 128,
                                            'sharing' : False,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 32}
        
        partitions['agate']['a100-4']  = {'cores_per_node' : 64,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 4}
        
        partitions['agate']['a100-8']  = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 7.5, # GB
                                            'max_nodes' : 4}
        
        partitions['mesabi'] = {}
        partitions['mesabi']['amdsmall'] = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 7.5, # GB
                                            'max_nodes' : 4}
    @property
    def options(self):
        options = self.base_options
        
        if not options['account']:                
            if self.machine in self.accounts:
                account = self.accounts[self.machine]
            else:
                raise NotImplementedError('Machine not supported')
        machine = self.machine
        
    @property
    def vasp_dir(self):
        if self.machine in self.msi_machines:
            self.vasp_dir = '/home/cbartel/shared/bin/vasp/'
        else:
            raise NotImplementedError('Machine not supported')