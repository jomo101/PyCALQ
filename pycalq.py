import logging

import general.task_manager as tm
import general.config_handler as ch
import general.project_directory as pd

import fvspectrum.sigmond_project_handler as sph
import fvspectrum.sigmond_view_corrs
import fvspectrum.sigmond_average_corrs
import fvspectrum.sigmond_rotate_corrs
import fvspectrum.sigmond_spectrum_fits
# import fvspectrum.generate_toy_correlators
import fvspectrum.compare_sigmond_levels
import QC2.single_channel_fit_mean
import QC2.run_HPW_fit

# Thanks to Drew and https://stackoverflow.com/a/48201163/191474
#ends code when run logging.error(message) or logging.critical(message)
# logging.warning(message) and logging.debug(message) won't end code but will output at certain verbosity settings
class ExitOnExceptionHandler(logging.StreamHandler):

  def emit(self, record):
    super().emit(record)
    if record.levelno in (logging.ERROR, logging.CRITICAL):
      raise RuntimeError
    
#set up logger
logging.basicConfig(format='%(levelname)s: %(message)s', handlers=[ExitOnExceptionHandler()], level=logging.INFO)

# set up the tasks config and order and classes
DEFAULT_TASKS = { #manage default configurations
    "tasks":
        {
            tm.Task.preview_corrs: None,
            tm.Task.average_corrs: None,
            tm.Task.rotate_corrs: None,
            tm.Task.fit_spectrum: None,
            # tm.Task.toy_corrs: None,
            tm.Task.compare_spectrums: None,
            tm.Task.single_channel_fit: None,
        }
}
                    
#manage order of tasks  
TASK_ORDER = list(dict(tm.Task.__members__).values())   
TASK_NAMES = {task.name:task for task in TASK_ORDER}
SIGMOND_TASKS = [ #manage which classes to use for each unique task -> change for selection (fvspectrum)
    tm.Task.preview_corrs,
    tm.Task.average_corrs,
    tm.Task.rotate_corrs,
    tm.Task.fit_spectrum,
    # tm.Task.toy_corrs,
]                            
TASK_MAP = { #manage which classes to use for each unique task -> change for selection (fvspectrum)
    tm.Task.preview_corrs: fvspectrum.sigmond_view_corrs.SigmondPreviewCorrs,
    tm.Task.average_corrs: fvspectrum.sigmond_average_corrs.SigmondAverageCorrs,
    tm.Task.rotate_corrs: fvspectrum.sigmond_rotate_corrs.SigmondRotateCorrs,
    tm.Task.fit_spectrum: fvspectrum.sigmond_spectrum_fits.SigmondSpectrumFits,
    # tm.Task.toy_corrs: fvspectrum.generate_toy_correlators.GenerateToyCorrs,
    tm.Task.compare_spectrums: fvspectrum.compare_sigmond_levels.CompareLevels,
    tm.Task.single_channel_fit: QC2.run_HPW_fit.HPWFitTask,
}
TASK_DOC = { #imports documentation from each task
    tm.Task.preview_corrs: fvspectrum.sigmond_view_corrs.doc,
    tm.Task.average_corrs: fvspectrum.sigmond_average_corrs.doc,
    tm.Task.rotate_corrs: fvspectrum.sigmond_rotate_corrs.doc,
    tm.Task.fit_spectrum: fvspectrum.sigmond_spectrum_fits.doc,
    # tm.Task.toy_corrs: fvspectrum.generate_toy_correlators.doc,
    tm.Task.compare_spectrums: fvspectrum.compare_sigmond_levels.doc,
    tm.Task.single_channel_fit: QC2.run_HPW_fit.doc,
}

#set required general parameters 
#items in list must be str or {str: list of str}
#we don't really want to go deeper than two levels
REQUIRED_GENERAL_CONFIGS = [
   'project_dir',
   'ensemble_id',
#    {'ensemble_info':['ensemble_id']},
]


class PyCALQ:

    def __init__( self, general_configs, task_configs = DEFAULT_TASKS ):
        #set up config handler
        self.general_configs_handler = ch.ConfigHandler(general_configs)
        self.task_configs_handler = ch.ConfigHandler(task_configs)

        #perform checks on configs
        self.general_configs_handler.check('general',REQUIRED_GENERAL_CONFIGS)
        self.task_configs_handler.check('tasks',[])
        
        self.general_configs = self.general_configs_handler.configs['general']
        self.task_configs = self.task_configs_handler.configs['tasks']

        if type(self.task_configs)!=list:
            logging.error("Error in task configuration file. List tasks.")

        self.proj_dir = pd.ProjectDirectoryHandler(self.general_configs['project_dir'],TASK_ORDER)

        #sort tasks by task order
        task_names = [task.name for task in TASK_ORDER]
        self.task_configs.sort(key = lambda x: task_names.index(list(x.keys())[0]) if list(x.keys())[0] in task_names else len(self.task_configs))

        #set up sigmond handler and set up project directory handler 
        max_task = 0
        self.sig_proj_hand = None
        sigmond_tasks = []
        for task in self.task_configs:
            key = list(task.keys())
            if len(key)>1:
                logging.error("Error in task configuration file.")
            if key[0] in TASK_NAMES.keys():
                if TASK_NAMES[key[0]] in SIGMOND_TASKS:
                    sigmond_tasks.append(TASK_NAMES[key[0]])
                max_task = task_names.index(key[0]) if task_names.index(key[0])>max_task else max_task
        if sigmond_tasks:
            self.sig_proj_hand = sph.SigmondProjectHandler(self.general_configs,sigmond_tasks)

        for task in TASK_ORDER[:max_task+1]:
            self.proj_dir.set_task(task.value, task.name)

    def run( self ):
        #perform the tasks in TASK_ORDER
        for task_config in self.task_configs:
            task_name = list(task_config.keys())[0] #root
            if task_name in TASK_NAMES:
                task = TASK_NAMES[task_name]
                self.proj_dir.set_task(task.value, task.name)

                #print documentation if requested
                if 'info' in task_config[task.name]:
                   if task_config[task.name]['info']:
                      print(TASK_DOC[task])

                #if sigmond task, set up project handler
                logging.info(f"Setting up task: {task.name}...")
                if task in SIGMOND_TASKS:
                    this_task = TASK_MAP[task](task.name, self.proj_dir, self.general_configs,task_config[task.name], self.sig_proj_hand) #initialize
                else:
                    this_task = TASK_MAP[task](task.name, self.proj_dir, self.general_configs,task_config[task.name]) #initialize
                logging.info(f"Task {task.name} set up.")

                #perform the task, produce the data
                logging.info(f"Running task: {task.name}...")
                this_task.run() 
                logging.info(f"Task {task.name} completed.")

                #do not plot if "plot: False" is present in task yaml
                if 'plot' in task_config[task.name]:
                    if not task_config[task.name]['plot']:
                        if task in SIGMOND_TASKS:
                            self.sig_proj_hand.switch_tasks()
                        continue
                   
                logging.info(f"Plotting task: {task.name}...")
                this_task.plot() 
                logging.info(f"Task {task.name} plotted.")
                
                #handle memory of sigmond tasks
                if task in SIGMOND_TASKS:
                   self.sig_proj_hand.switch_tasks()





