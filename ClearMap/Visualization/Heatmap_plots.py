import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import matplotlib
import math
#important for text to be detected when importing saved figures into illustrator
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
plt.rcParams.update({'font.size': 22})

# returns the paths to the raw csv files, names of the output pre-processed csv files with 
# level information, and the conditions of the raw csv files in correct order for later processes
def LoadMetaData(meta_info_path):
    meta_info_df = pd.read_csv(meta_info_path)
    datapaths = [] # raw csv path
    depth_df_paths = [] # pre-processed csv paths/output of finding levels, input for heatmap plot
    conditions = [] # conditions of each raw csv
    subjects = []
    meta_info_df.drop(columns = ['Designation', 'Status', 'Notes', 'Analysis', 'Done'])
    for i in range(len(meta_info_df)):
        datapaths.append(meta_info_df[i, 1])
        depth_df_paths.append(meta_info_df[i, 0] + '_levels.csv')
        conditions.append(meta_info_df[i, 2])
        subjects.append(meta_info_df[i, 0])
    return datapaths, depth_df_paths, conditions

# load the atlas in using its path
def LoadingAtlas(brain_atlas_annotationpath):
    brain_atlas_annotationdf = pd.read_csv(brain_atlas_annotationpath,index_col=False)

    return brain_atlas_annotationdf

# load the original dataset in using its datapath
def LoadingDf(datapath):
    # load and preprocess the data
    total_merge_df = pd.read_csv(datapath)
    total_merge_df.fillna("nan",inplace=True)
    total_merge_df = total_merge_df.sort_values(by=['id'])
    total_merge_df = total_merge_df.reset_index(drop=True)
    for ind in total_merge_df.index:
        total_merge_df['Name'][ind] = total_merge_df['Name'][ind].strip()

    return total_merge_df

# removes ventricles and other regions that are not present in the brain atlas
def TrimmingDf(total_merge_df, brain_atlas_annotationpath, user_defined_regions = None):
    if user_defined_regions == None:
        brain_atlas_annotationdf = LoadingAtlas(brain_atlas_annotationpath)
        subset = brain_atlas_annotationdf.acronym.values
        total_merge_df = total_merge_df[total_merge_df['acronym'].isin(subset)] 
    else: 
        total_merge_df = total_merge_df[total_merge_df['acronym'].isin(user_defined_regions)] 

    return total_merge_df

# this method loads in a csv found in datapath and recursively finds
# the level of each brain region in the csv. It outputs this modified csv to depth_df_path
def FindingLevels(datapaths, brain_atlas_annotationpath, depth_df_paths):
    for i in range(len(datapaths)):
        datapath = datapaths[i]
        depth_df_path = depth_df_paths[i]
        total_merge_df = LoadingDf(datapath)
        total_merge_df['level'] = total_merge_df.counts
        #iterate over all the existing rows
        for i in total_merge_df.id:
            level = 0

            # the index of the row we are analyzing
            current = total_merge_df.index[total_merge_df['id'] == i].tolist()
            # initial parentID of row being analyzed
            parentID = total_merge_df.iloc[current[0]]['parent_id']
            i = parentID

            # if the parentID is not null, we look through the parents of the parents etc.
            while (parentID != 'nan'):

                # find the parentID of the row that has the id == i
                boom = total_merge_df.index[total_merge_df['id'] == i].tolist()
                parentID = total_merge_df.iloc[boom[0]]['parent_id']

                # we set the id to the parentID and look at that row, we add a level each time we have to do this
                i = parentID
                level = level+1
            # add this final result to our dataframe
            total_merge_df['level'][current[0]] = level

        total_merge_df = total_merge_df[['id','counts','newcounts','Name','structure_order','parent_id','parent_acronym','acronym', 'level']]
        total_merge_df = TrimmingDf(total_merge_df, brain_atlas_annotationpath)

        # saves the modified csv with all the depth information to outputpath
        total_merge_df.to_csv(depth_df_path, index = False)

# this method identifies the ancestor of a desired default_depth of all brain regions
# of the indicated depth. It does this for the csv located at depth_df_path
def FindingParent(depth_df_path, deeper_depth, default_depth):
    # might want to change this because it is where we get our data from. This line is simply pointing to 
    # a path that a previous method has exported to but we might want to reference this previous method directly
    total_merge_df = pd.read_csv(depth_df_path, index_col = False)
    #total_merge_df[total_merge_df.level == 1]
    # change to reformated_df or something like that

    total_merge_df['default_depth'] = "nan"

    # gives the name of the region at default_depth for subregions of indicated deeper_depth
    for i in total_merge_df.id:
        current = total_merge_df.index[total_merge_df['id'] == i].tolist()
        current_level = total_merge_df.iloc[current[0]]['level']
        if (current_level == deeper_depth):
            while (current_level != default_depth):
                boom = total_merge_df.index[total_merge_df['id'] == i].tolist()
                parentID = total_merge_df.iloc[boom[0]]['parent_id']
                current_level = total_merge_df.iloc[boom[0]]['level']
                current_acr = total_merge_df.iloc[boom[0]]['acronym']
                i = parentID
            total_merge_df['default_depth'][current[0]] = current_acr

    return total_merge_df

# this method takes all the children of an indicated deeper_depth of only the desired_region
# that is at indicated default_depth. This is done for the csv located at depth_df_path
def ProcessChild(depth_df_path, deeper_depth, default_depth, desired_region):
    total_merge_df = FindingParent(depth_df_path, deeper_depth, default_depth)
    
    # drops rows that are not of the desired level
    total_merge_df = total_merge_df.drop(total_merge_df[total_merge_df['level'] != deeper_depth].index)
    total_merge_df = total_merge_df[total_merge_df.default_depth == desired_region].reset_index(drop = True)
    total_merge_df.sort_values(by=['default_depth'])

    if total_merge_df.empty:
        raise ValueError("The indicated region (" + desired_region +") that you want subregions of has no subregions at the level indicated by \"deeper_depth\". Please \
                         choose another region whose subregions you wish to look at or potentially a more shallow \"deeper_depth\". ")
    
    names = total_merge_df["acronym"].tolist()
    names = [str(name) for name in names]

    return total_merge_df, names

# this method discards all brain regions that are not at the same depth as default_depth
# allows us to first locate all of the regions of default_depth and then we can do further
# analysis on its children using the ProcessChild method
def ProcessParent(depth_df_path, deeper_depth, default_depth):
    total_merge_df = FindingParent(depth_df_path, deeper_depth, default_depth)
    
    # drops rows that are not of the desired level
    total_merge_df = total_merge_df.drop(total_merge_df[total_merge_df['level'] != default_depth].index)
    total_merge_df.sort_values(by=['default_depth'])

    names = total_merge_df["acronym"].tolist()
    names = [str(name) for name in names]

    return total_merge_df, names

# this method combines multiple csv's that have already done the previous processing steps
# each of the csv's in depth_df_paths represent a different condition. If desired_region is None
# then it uses ProcessParent and assumes you only want the regions are default_depth level
# If you input a desired_region then it will go through ProcessChild and give you the brain 
# regions at level: 'deeper_depth' that are children/grandchildren/etc. of the specific desired_region indicated
def AddingConditions(depth_df_paths, deeper_depth, default_depth, conditions, subjects, desired_region):
    data = []
    total_heatmap_df = pd.DataFrame(data) 
    for iteration, depth_df_path in enumerate(depth_df_paths):
        if desired_region == None:
            current_df, names = ProcessParent(depth_df_path, deeper_depth, default_depth)
        else:
            current_df, names = ProcessChild(depth_df_path, deeper_depth, default_depth, desired_region)

        current_df['conditions'] = conditions[iteration]
        current_df['subject']  = subjects[iteration]
        total_heatmap_df = total_heatmap_df.append(current_df, ignore_index=False)
    
    return total_heatmap_df, names

# computes a mean and standard deviation of a row based on the names of the condition given in 
# control_condition. If none are given then we simply take all of the conditions
# we then calculate the z-score for every data point in a row based on this mean and SD
def Zscore(pivoted_df, control_condition = None):
    num_cond = pivoted_df.shape[1] - 1
    if control_condition == None:
        control_condition = np.arange(num_cond)
    
    # we enter a row of the dataframe
    for ind in pivoted_df.index:
        # we calculate the mean and SD of that row
        i=0
        zscored = np.zeros(num_cond)
        base = np.zeros(len(control_condition))
        for condition in control_condition:
            base[i] = pivoted_df[condition][ind]
            i = i+1
        mean = np.mean(base)
        sd = np.std(base)

        for i in range(num_cond):
            zscored[i] = float(pivoted_df[i][ind] - mean) / sd
            pivoted_df[i][ind] = zscored[i] 

    return pivoted_df

# calculates the mean and SD of each row and return an array with these values
def MeanCalc(pivoted_single_cond_df):
    stats = np.zeros(shape = (len(pivoted_single_cond_df), 2))
    for i in range(len(pivoted_single_cond_df)):
        values = np.zeros(len(pivoted_single_cond_df.columns)-1)
        for j in range(len(pivoted_single_cond_df.columns)-1):
            values[j] = pivoted_single_cond_df.iloc[i, j]
        stats[i, 0] = np.mean(values)
        stats[i, 1] = np.std(values)
    
    return stats

def MeanCalcPrePivot(total_heatmap_df, control_condition, num_cond):
    keep_vals = np.zeros(shape = (len(total_heatmap_df)/num_cond, num_cond))
    stats = np.zeros(shape = (len(total_heatmap_df)/num_cond, 2))
    for cond in range(num_cond):
        for i in range(total_heatmap_df.index):
            # newcounts: col 0
            # acronym: col 1
            # condition: col 2
            # subject: col 3
            #make sure you are referring to the appropriate columns: 0 and 4 are random tbh so check it
            if (total_heatmap_df[i, 2] == control_condition):
                keep_vals = total_heatmap_df[i, 0]

def PlotFunc(total_heatmap_df, heatmap_plot_output_path, conditions, names, num_cond, control_condition, figsize, max, min = 0, zscore = False):
    fig,axs = plt.subplots(1, len(set(conditions)), sharey = True, figsize = figsize)
    stats = None
    if zscore:
        heatmap_plot_output_path = heatmap_plot_output_path + '_zscored'

    for iter, cond in enumerate(set(conditions)):
        # isolates the condition from this iteration
        single_cond_df = total_heatmap_df.copy()
        single_cond_df = single_cond_df[single_cond_df.conditions == cond].reset_index(drop = True)
        # original df pivoted into heatmap form (change variable names), this could be called heatmap_df
        pivoted_single_cond_df = single_cond_df.pivot(index="acronym", columns="subject", values="newcounts")

        # necessary for calculating the zscore df. We do the calculations here because we are already
        # parsing through and its much easier (and faster) to do it on the pivoted df
        if (cond == control_condition) and (zscore == False):
            stats = MeanCalc(pivoted_single_cond_df)

        #plotting
        axs[iter].set_title("Condition " + str(cond))
        cbar = False
        if iter == (len(set(conditions)) - 1): 
            cbar = True
        g = sns.heatmap(data = pivoted_single_cond_df, ax = axs[iter], vmin = (min - max*0.2 - 3), vmax = (max*1.2 + 3), linewidths = 0.4, linecolor = 'white', cbar = cbar)
        g.set_yticks(np.linspace(0.5, len(names)+0.5, num = len(names)+1)[0:-1])
        g.set_yticklabels(labels = names, rotation = 0) 
        plt.xticks(rotation=0)

    fig.savefig(heatmap_plot_output_path)

    return stats

# plots the heatmap indicated by the final csv with all the conditions
# depth_df_paths is a list of the filepaths+names of the preprocessed csv's
# deeper_depth is the level of the children and default_depth is the level of the initial heatmap
# heatmap_plot_output_path indicates the path of the output heatmap figures
# If no desired region is given, then it will display the heatmap for all of its children at the level indicated by deeper_depth
# If no conditions are given everything is assumed to have the same condition. Must be passed as a list corresponding to 
# depth_df_paths with each entry corresponding to a condition number
# The control condition is the one off which we base the mean and SD calculations for zscoring. If no control_condition is
# given then we do not give a zscored heatmap
def PlotHeatmap(depth_df_paths, deeper_depth, default_depth, heatmap_plot_output_path, conditions = None, 
                subjects = None, desired_region = None, control_condition = None, figsize = (20, 55), **kwargs):
    
    # makes everything condition 0 if no conditions are given and gives the number of different conditions
    if conditions == None:
        conditions = np.zeros(len(depth_df_paths))
    if subjects == None:
        subjects = range(len(depth_df_paths))

    # defining some helpful variables
    num_cond = len(set(conditions))
    num_csv = len(depth_df_paths)

    # adds the condition and subject information to our dataframes and combines them into one large df
    total_heatmap_df, names = AddingConditions(depth_df_paths, deeper_depth, default_depth, conditions, subjects, desired_region)

    # get max value of cell counts for heatmap labelling
    max = np.nanmax(total_heatmap_df.newcounts.values)

    # we perform this on the massive df and remove unnecessary columns and null rows
    total_heatmap_df = total_heatmap_df.drop(columns = ['id', 'counts', 'structure_order', 'parent_id', 'parent_acronym', 'Name', 'level'])
    total_heatmap_df.loc[total_heatmap_df['newcounts'] == "nan", 'newcounts'] = 0.0
    total_heatmap_df[['newcounts']] = total_heatmap_df[['newcounts']].apply(pd.to_numeric)

    # iterates through the different conditions, subsetting the large dataframe to only plot one condition at a time
    stats = PlotFunc(total_heatmap_df, heatmap_plot_output_path, conditions, names, num_cond, control_condition, figsize, max)

    # if we are given a control_condition, we then zscore all of the data points around the 
    # mean and SD of that given condition and plot it
    if not (control_condition == None):
        total_heatmap_df.reset_index(inplace = True)
        max = 0
        min = 0
        for ind in total_heatmap_df.index:
            zscore = float((total_heatmap_df['newcounts'][ind] - stats[ind/num_csv][0])) / stats[ind/num_csv][1]
            if zscore > max:
                max = zscore
            if zscore < min:
                min = zscore
            total_heatmap_df['newcounts'][ind] = zscore

        PlotFunc(total_heatmap_df, heatmap_plot_output_path, conditions, names, num_cond, control_condition, figsize, max, min = min, zscore = True)

# make this a list or numpy file containing brain regions
brain_atlas_annotationpath = r"S:\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\atlas_info_KimRef_FPbasedLabel_v2.9_brain.csv"
meta_info = r"\\128.95.12.251\LabCommon\Ken\data\David_TRAP_data\meta\metainfo.csv"
#datapaths, depth_df_paths, conditions, subjects = LoadMetaData(meta_info)
datapaths = [r"Data/CFRL_Annotated_counts_clean.csv", r"Data/CFRN_Annotated_counts_clean.csv", r"Data/CFRR_Annotated_counts_clean.csv", 
            r"Data/HFNL_Annotated_counts_clean.csv", r"Data/HFNN_Annotated_counts_clean.csv", r"Data/HMR1_Annotated_counts_clean.csv", 
            r"Data/HMR2_Annotated_counts_clean.csv", r"Data/HFRR_Annotated_counts_clean.csv", r"Data/HFRL_Annotated_counts_clean.csv", 
            r"Data/HFRN_Annotated_counts_clean.csv"]
depth_df_paths = [r"sample_output_levels2.csv", r"sample_output_levels3.csv", r"sample_output_levels4.csv", r"sample_output_levels5.csv", 
                 r"sample_output_levels6.csv", r"sample_output_levels7.csv", r"sample_output_levels8.csv", r"sample_output_levels9.csv", 
                 r"sample_output_levels10.csv", r"sample_output_levels11.csv"]
deeper_depth = 6
default_depth = 4
heatmap_plot_output_path = 'output_heatmap_figure_conditions_children'
subjects = ["CFRL", "CFRN", "CFRR", "HFNL", "HFNN", "HFRL", "HFRN", "HFRR", "HMR1", "HMR2"]
conditions = (0, 0, 0, 1, 1, 1, 2, 2, 2, 2)
#FindingLevels(datapaths, brain_atlas_annotationpath, depth_df_paths)
PlotHeatmap(depth_df_paths, deeper_depth, default_depth, heatmap_plot_output_path, conditions, subjects, desired_region = "HY", control_condition = 0)