
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
        partitions['agate'] = {''}
    @property
    def options(self):
        options = self.base_options
        
        if not options['account']:                
            if self.machine in self.accounts:
                account = self.accounts[self.machine]
            else:
                raise NotImplementedError('Machine not supported')
        machine = self.machine
        
        if machine == 'agate'
    @property
    def vasp_dir(self):
        if self.machine in self.msi_machines:
            self.vasp_dir = '/home/cbartel/shared/bin/vasp/'
        else:
            raise NotImplementedError('Machine not supported')