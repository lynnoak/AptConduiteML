from feature_extraction.DatabaseConnection import myDatabase
from bson import ObjectId, errors

from pathlib import Path
import pandas as pd
import numpy as np
import math
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-GUI backend


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


TIME_COLLECTION = {
    'scenario_1':['sc_time_list','sc_brake_list'],
    'scenario_2':['sc_time_list','sc_brake_list','sc_command_list'],
    'scenario_3':['sc_time_list','sc_brake_list','sc_command_list','sc_throttle_list','sc_distance_list','sc_vit_vehicule_suiveur_list','sc_vit_vehicule_suivi_list'],
    'scenario_4':['sc_time_list','sc_brake_list','sc_steer_list','sc_throttle_list']
}


TASK_COLLECTION = {
    'scenario_1':['sc_reaction_time','sc_time_apparition','sc_ball_position'],
    'scenario_2':['sc_reaction_time','sc_time_apparition','sc_road_sign_coord','sc_stimulii','sc_ordre_env','sc_resultat_list'],
    'scenario_3':['sc_reaction_time','sc_time_apparition','sc_etoile_list','sc_collision_counter','sc_commodo_err_counter'],
    'scenario_4':['sc_collision']
}

#Reaction time threshold which too short a reaction time should be identified as an error
TOO_SHORT = 0.001
# Distance threshold which the follower car should keep near than it
TOO_FAR = 90
TOO_CLOSE = 7
SIM_ALL_SC = ['Reaction', 'Brake', 'Performance']

# mapping value to score
def value_to_score(value, max_task, column_name):
    if not isinstance(value, (int, float)):
        return np.nan

    # correct counts → linear normalization
    if column_name.startswith('Num_cor'):
        return value / max_task

    # error counts → exponential penalty 
    if 'err' in column_name or 'error' in column_name:
        return math.exp(-(math.log(1000) / max_task) * value)

    # everything else is NOT part of performance score
    return np.nan



class DataSim:
    """
    Data Collected from physiological sensors
    parameters:
        scenario_n: the type ID of the scenario being queried
        sc_scenario: the id of the scenario being queried
    """
    # Subclass registry
    _registry = {}

    @classmethod
    def register_scenario(cls, scenario_n):
        """Decorator to register subclasses"""
        def wrapper(subclass):
            cls._registry[scenario_n] = subclass
            return subclass
        return wrapper

    def __init__(self, scenario_n, sc_scenario,save_path = None):
        
        # Dynamically create subclass instances based on scenario_n
        if scenario_n not in self._registry:
            raise ValueError("Unsupported scenario")
        else:
            self.scenario_n = scenario_n

        if type(self) is DataSim:
            self._instance = self._registry[scenario_n](scenario_n, sc_scenario, save_path)
            self.__dict__.update(self._instance.__dict__)
        else:
            try:
                self.sc_scenario = ObjectId(sc_scenario)
            except errors.InvalidId:
                raise ValueError(f"Invalid ObjectId: {sc_scenario}")
            
            self.db_client = myDatabase()

            self.data = self.db_client.get_collection(self.scenario_n,query = {"_id": self.sc_scenario})

            if self.data.empty:
                raise ValueError(f"The queried scenario dataset does not contain {sc_scenario}")

            pat = self.data['sc_pat'].iloc[0]
            self.pat = str(pat)

            if not save_path:
                current_date = datetime.now().strftime("%Y-%m-%d")
                self.save_path = Path("./tempSim") / str(current_date)/ str(scenario_n) / str(sc_scenario)
            else:
                self.save_path = save_path
            self.save_path.mkdir(parents=True, exist_ok=True)


            config = pd.Series(self.data['sc_config'].iloc[0])
            config = pd.concat([config, pd.Series(self.data[['sc_ecg','sc_eda','sc_eeg','sc_eyetracker','sc_headtracker']].iloc[0])])
            config.to_csv(self.save_path/"config.csv")
            self.config = config

            # extract the time length data
            time_array = self._load_data(TIME_COLLECTION[self.scenario_n])
            print(f'There are {time_array.shape[0]} rows of time-length data for {sc_scenario}. ')
            if scenario_n =='scenario_1' and time_array.shape[0] != 0:
                time_array['sc_brake_list'] = 1-time_array['sc_brake_list']
            time_array.to_csv(self.save_path/"time_array.csv")
            self.time_array = time_array
            
            # extract the task length data 
            task_array = self._load_data(TASK_COLLECTION[self.scenario_n])
            print(f'There are {task_array.shape[0]} rows of task-length data for {sc_scenario}. ')
            task_array.to_csv(self.save_path/"task_array.csv")
            self.task_array = task_array

            self.feature = pd.Series(dtype=float)

    def _load_data(self, collection_list):
        data_array = pd.DataFrame()  
        
        if not collection_list: 
            return data_array

        data_array = {}
        array_length = 0
        
        for collection_name in collection_list:

            df = self.db_client.get_collection(collection_name,query = {"sc_scenario": self.sc_scenario})
            
            if not df.empty:  
                df.drop(columns=['_id', 'sc_scenario'], inplace=True, errors='ignore') 
                for col in df.columns:
                    t = df[col].iloc[0]
                    if isinstance(t, list):
                        data_array[col] = t
                        if array_length == 0:
                            array_length = len(t)
                        elif array_length == len(t):
                            continue
                        else:
                            return pd.DataFrame()
                    else:
                        data_array[col] = [t for i in range(array_length)]
            else:
                print(f'There is no data loaded from {collection_name} for {self.sc_scenario}')


        data_array = pd.DataFrame(data_array)

        return data_array

    def process(self,show_plot = False):

        if self.time_array.shape[0] ==0 or self.task_array.shape[0] == 0:
            print(f"Missing Data for {self.sc_scenario}")
            return
        
        try:
            self._instance.process(show_plot)
            self.__dict__.update(self._instance.__dict__)
        except Exception as e:
            print(e)   

        #print(self.scenario_n)
        #print(self.feature)

        for col in SIM_ALL_SC:
            if col not in self.feature:
                self.feature[col] = np.nan

        self.feature = self.feature.to_frame().T

        # Add SIM_ prefix to all feature names
        self.feature = self.feature.add_prefix("SIM_")


        self.feature.to_csv(self.save_path / 'featureSimSC.csv', index=False)
        self.feature[[f"SIM_{c}" for c in SIM_ALL_SC]].to_csv(
            self.save_path / 'featureSim.csv',
            index=False
        )
    
    def _interface_stat(self,interface,show_plot = False):

        time_array  = self.time_array.copy()
        inferface_list = f'sc_{interface}_list'
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_array['sc_time_list'], time_array[inferface_list], marker='o', linestyle='-')
        plt.title('Value Over Time')
        plt.xlabel('Time')
        plt.ylabel(interface)
        plt.grid(True)
        plt.savefig(self.save_path/str(interface))

        if show_plot:
            plt.show(block=True)
        
        plt.close('all')

        # find the events
        time_array['event'] = (time_array[inferface_list] > 0).astype(int).diff().fillna(0)
        time_array['event_start'] = (time_array['event'] == 1).cumsum()
        time_array = time_array[time_array[inferface_list] > 0]

        # If no active events, return zeros
        if time_array.empty or 'event_start' not in time_array.columns:
            return 0, pd.DataFrame({'max':[0], 'min':[0], 'mean':[0], 'std':[0]})

        count = time_array['event_start'].max()
        stats = time_array.groupby('event_start')[inferface_list].agg(['max', 'min', 'mean', 'std'])
        return count, stats

# Register subclasses
@DataSim.register_scenario('scenario_1')
class DataSimScenario1(DataSim):

    def process(self, show_plot=False):

        brake_count, brake_stats = self._interface_stat(
            interface='brake',
            show_plot=show_plot
        )

        task_array = self.task_array.copy()

        reaction_col = task_array['sc_reaction_time'].copy()

        max_num = int(self.config['conf_nombre_sphere'])
        max_duree = float(self.config['conf_duree'])

        performance_values = pd.DataFrame()

        # Correct reaction mean (only correct ones)
        self.feature['Rea_cor'] = (
            reaction_col[reaction_col > TOO_SHORT].mean()
            if not reaction_col.empty
            else max_duree
        )

        # Missed reactions
        self.feature['Num_miss'] = reaction_col[reaction_col < 0].count()

        performance_values.at['Num_cor', 'value'] = (
            reaction_col[reaction_col > TOO_SHORT].count()
        )
        performance_values.at['Num_cor', 'max'] = max_num

        performance_values.at['Num_error', 'value'] = max(
            brake_count - max_num, 0
        )
        performance_values.at['Num_error', 'max'] = max_num

        # Correct count per position
        stats = task_array.groupby('sc_ball_position')['sc_reaction_time'].apply(
            lambda x: (x > TOO_SHORT).sum()
        ).reset_index(name='Num_cor_pos')

        pos_col = []

        for _, row in stats.iterrows():
            pos_id = row['sc_ball_position']
            pos_label = f'Num_cor_pos_{pos_id}'
            performance_values.loc[pos_label, 'value'] = row['Num_cor_pos']
            pos_col.append(pos_label)

        if len(pos_col) > 0:
            number_max_pos = int(max_num / len(pos_col))
            performance_values.loc[pos_col, 'max'] = number_max_pos

        # -------------------------------------------------
        # NEW: Correct reaction mean per position
        # Defensive rule:
        #   - >=2 correct → mean
        #   - 0 or 1 correct → max_duree
        # -------------------------------------------------
        reaction_stats = task_array.groupby('sc_ball_position')['sc_reaction_time'].apply(
            lambda x: (
                x[x > TOO_SHORT].mean()
                if (x > TOO_SHORT).sum() >= 2
                else max_duree
            )
        ).reset_index(name='Rea_cor_pos')

        for _, row in reaction_stats.iterrows():
            pos_id = row['sc_ball_position']
            pos_label = f'Rea_cor_pos_{pos_id}'
            self.feature[pos_label] = row['Rea_cor_pos']

        # Compute performance score
        performance_values['score'] = performance_values.apply(
            lambda row: value_to_score(
                row['value'],
                row['max'],
                row.name
            ),
            axis=1
        )

        performance_score = pd.to_numeric(
            performance_values['score'],
            errors='coerce'
        ).dropna()

        # Save raw count features
        for k, v in performance_values['value'].items():
            self.feature[k] = v

        # General features
        reaction_col = reaction_col.where(
            reaction_col >= TOO_SHORT,
            max_duree
        )

        self.feature['Reaction'] = reaction_col.mean()

        self.feature['Brake'] = (
            brake_stats['std'].mean()
            if not brake_stats.empty
            else 0
        )

        count_score = (
            performance_score.mean()
            if not performance_score.empty
            else 0
        )

        rea_score = math.exp(
            -(math.log(1000) / max_duree) * self.feature['Rea_cor']
        )

        self.feature['Performance'] = 0.7 * count_score + 0.3 * rea_score

        self.feature['Performance'] = float(
            np.clip(self.feature['Performance'], 0, 1)
        )


# Register subclasses
@DataSim.register_scenario('scenario_2')
class DataSimScenario2(DataSim):
    def process(self,show_plot = False):

        brake_count,brake_stats = self._interface_stat(interface ='brake', show_plot= show_plot )
        command_count,command_stats = self._interface_stat(interface ='command', show_plot= show_plot)
        task_array = self.task_array.copy()# Avoid modifying the original data

        # Ensure `sc_resultat_list` column is of string type
        task_array['sc_resultat_list'] = task_array['sc_resultat_list'].astype(str)
        task_array['sc_ordre_env'] = task_array['sc_ordre_env'].astype(str)

        reaction_col = task_array['sc_reaction_time'].copy()
        max_num = int(self.config['conf_nb_total_stimuli'])
        max_duree = float(self.config['conf_duree_totale'])
        performance_values = pd.DataFrame()

        # Filter correct responses (reaction time > threshold and result is '1' or '4')
        correct_col = task_array[(task_array['sc_reaction_time'] > TOO_SHORT) & task_array['sc_resultat_list'].isin(['1', '4'])]
        error_col = task_array[task_array['sc_resultat_list'].isin(['2', '5'])]

        # Calculate the number of correct responses and their average reaction time
        performance_values.at['Num_cor','value']  = correct_col.shape[0]  # Count the number of rows
        performance_values.at['Num_cor','max'] = max_num
        self.feature['Rea_cor'] = correct_col['sc_reaction_time'].mean() if not correct_col.empty else max_duree # Compute the mean of numerical columns

        # Calculate the number of missed responses
        self.feature['Num_miss'] = task_array[task_array['sc_resultat_list'].isin(['3', '6'])].shape[0]

        # Calculate the numbers of specific error types
        performance_values.at['Num_error_sign','value']  = error_col.shape[0]
        performance_values.at['Num_error_sign','max'] = max_num
        self.feature['Rea_error_sign'] = error_col['sc_reaction_time'].mean() if not error_col.empty else 0

        performance_values.at['Num_error','value']  = performance_values.at['Num_error','value'] = max((brake_count + command_count) - max_num, 0)

        performance_values.at['Num_error','max'] = max_num

        performance_values.at['Num_err_2_for_1','value']  = (task_array['sc_resultat_list'] == '2').sum()
        performance_values.at['Num_err_2_for_1','max'] = max_num/2
        performance_values.at['Num_err_1_for_2','value']  = (task_array['sc_resultat_list'] == '5').sum()
        performance_values.at['Num_err_1_for_2','max'] = max_num/2

        #Calculate the numbers in different environment
        performance_values.at['Num_cor_env_1','value']  = (correct_col['sc_ordre_env']== '1').sum()  
        performance_values.at['Num_cor_env_1','max'] = max_num/2
        performance_values.at['Num_cor_env_2','value']  = (correct_col['sc_ordre_env']== '-1').sum()  
        performance_values.at['Num_cor_env_2','max'] = max_num/2
        performance_values.at['Num_err_env_1','value']  = (error_col['sc_ordre_env']== '1').sum()  
        performance_values.at['Num_err_env_1','max'] = max_num/2
        performance_values.at['Num_err_env_2','value']  = (error_col['sc_ordre_env']== '-1').sum()  
        performance_values.at['Num_err_env_2','max'] = max_num/2

        #Compute the score for performance
        performance_values['score'] = performance_values.apply(
            lambda row: value_to_score(row['value'], row['max'], row.name), axis=1
        )
        performance_score = pd.to_numeric(performance_values['score'], errors='coerce')
        performance_score = performance_score.dropna()

        for k, v in performance_values['value'].items():
            self.feature[k] = v

        #print(performance_values)

        #Compute the general features
        reaction_col = reaction_col.where(reaction_col>=TOO_SHORT,max_duree)
        self.feature['Reaction'] = reaction_col.mean()
        self.feature['Brake'] = brake_stats['std'].mean()
        count_score = performance_score.mean() if not performance_score.empty else 0

        rea_score = math.exp(
            -(math.log(1000) / max_duree) * self.feature['Rea_cor']
        )

        self.feature['Performance'] = 0.7 * count_score + 0.3 * rea_score
        self.feature['Performance'] = float(
    np.clip(self.feature['Performance'], 0, 1)
)


# Register subclasses
@DataSim.register_scenario('scenario_3')
class DataSimScenario3(DataSim):
    def process(self,show_plot = False):
        brake_count,brake_stats = self._interface_stat(interface ='brake', show_plot= show_plot)
        command_count,command_stats = self._interface_stat(interface ='command', show_plot= show_plot )
        throttle_count,throttle_stats = self._interface_stat(interface ='throttle', show_plot= show_plot )
        distance_count,distance_stats = self._interface_stat(interface ='distance', show_plot= show_plot )
        task_array = self.task_array.copy()
        time_array = self.time_array.copy()

        reaction_col = self.task_array['sc_reaction_time'].copy()
        max_num = int(self.config['conf_nb_total_stimuli_peripherie'])
        max_duree = float(self.config['conf_temps_illumination'])
        performance_values = pd.DataFrame()

        # Correct reactions
        self.feature['Rea_cor'] = reaction_col[reaction_col > TOO_SHORT].mean() if not reaction_col.empty else max_duree
        self.feature['Num_miss'] = reaction_col[reaction_col<0].count()
        performance_values.at['Num_cor','value'] = reaction_col[reaction_col>TOO_SHORT].count()
        performance_values.at['Num_cor','max'] = max_num

        performance_values.at['Num_error','value'] = max(task_array['sc_commodo_err_counter'].iloc[0], 0)
        performance_values.at['Num_error','max'] = max_num

        performance_values.at['Num_error_coll','value'] = max(task_array['sc_collision_counter'].iloc[0], 0)
        performance_values.at['Num_error_coll','max'] = max_num

        # Compute the speed
        self.feature['Dis_control'] = time_array['sc_distance_list'].std()
        dis_close = time_array[time_array['sc_distance_list']< TOO_CLOSE]
        dis_far = time_array[time_array['sc_distance_list'] > TOO_FAR]
        performance_values.at['Dis_error_close','value'] = dis_close.shape[0]/time_array.shape[0]
        performance_values.at['Dis_error_close','max'] = 1
        performance_values.at['Dis_error_far','value'] = dis_far.shape[0]/time_array.shape[0]
        performance_values.at['Dis_error_far','max'] = 1

        #print(performance_values)

        #Compute the score for performance
        performance_values['score'] = performance_values.apply(
            lambda row: value_to_score(row['value'], row['max'], row.name), axis=1
        )
        performance_score = pd.to_numeric(performance_values['score'], errors='coerce')
        performance_score = performance_score.dropna()

        #print(performance_score)
        for k, v in performance_values['value'].items():
            self.feature[k] = v

        #Compute the general features
        reaction_col = reaction_col.where(reaction_col>=TOO_SHORT,max_duree)
        self.feature['Reaction'] = reaction_col.mean()
        self.feature['Brake'] = brake_stats['std'].mean()
        self.feature['Throttle'] = throttle_stats['std'].mean()
        count_score = performance_score.mean() if not performance_score.empty else 0

        rea_score = math.exp(
            -(math.log(1000) / max_duree) * self.feature['Rea_cor']
        )

        self.feature['Performance'] = 0.7 * count_score + 0.3 * rea_score
        self.feature['Performance'] = float(
    np.clip(self.feature['Performance'], 0, 1)
)


# Register subclasses
@DataSim.register_scenario('scenario_4')
class DataSimScenario4(DataSim):
    def process(self,show_plot = False):
        print("Processing additional features for scenario 4")
        brake_count,brake_stats = self._interface_stat(interface ='brake', show_plot= show_plot)
        throttle_count,throttle_stats = self._interface_stat(interface ='throttle', show_plot= show_plot )
        steer_count,steer_stats = self._interface_stat(interface ='steer', show_plot= show_plot )
        if 'sc_collision' in self.task_array.columns:
            num_collision = self.task_array['sc_collision'].count()
            self.feature['Num_collision'] = num_collision
            self.feature['Performance'] = math.exp(-0.1 * num_collision)
        else:
            self.feature['Num_collision'] = 0
            self.feature['Performance'] = 1
        
        self.feature['Brake'] = brake_stats['std'].mean()
        self.feature['Throttle'] = throttle_stats['std'].mean()
        self.feature['Steer'] = steer_stats['std'].mean()
        def reaction_proxy(count, stats):
            if stats.empty:
                return np.inf
            return (
                1 / (count + 1e-3) +
                (stats['max'].mean() - stats['min'].mean()) * 0.5 +
                stats['std'].mean()
            )

        brake_proxy = reaction_proxy(brake_count, brake_stats)
        throttle_proxy = reaction_proxy(throttle_count, throttle_stats)
        steer_proxy = reaction_proxy(steer_count, steer_stats)

        reaction_proxy_value = min(brake_proxy, throttle_proxy, steer_proxy)
        # Reaction (scenario 4):
        # proxy of control responsiveness during free driving
        # not task reaction time
        self.feature['Reaction_control_proxy'] = reaction_proxy_value
        self.feature['Reaction'] = reaction_proxy_value * 0.5  



#For test this file alone, should be deleted
if __name__ == "__main__":
    scenario_list = ['scenario_1','scenario_2','scenario_3','scenario_4']
    db = myDatabase()
    for scenario_n in scenario_list:
        df = db.get_collection(scenario_n)
        df["_id"] = df["_id"].astype(str)
        for sc_scenario in df['_id'][0:2]:
            data = DataSim(scenario_n,sc_scenario,save_path=Path("./tempSim") / str(scenario_n) /str(sc_scenario))
            data.process()
    db.close()

