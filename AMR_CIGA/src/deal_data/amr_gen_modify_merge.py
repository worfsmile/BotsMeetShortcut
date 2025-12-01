#save the resultant AMR graphs in the form of their Penman notation
import json

import glob
import amrlib
import pandas as pd
import penman
from penman import constant
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.rbw_aligner import RBWAligner
from penman.models.noop import NoOpModel
import ast
import pickle
import os

#************************************************************************************
# Adding dummy source node and COMMENT edges to multiple AMRs and merge into one.
#************************************************************************************

def add_edge(graph, source, role, target):

    """Function to add an edge between two previously existing nodes in the graph.
    
    Here, source and target node instances already exist in "graph" and we simply add an edge with relation "rel"
    between the two. The purpose of this is to add :COREF and :SAME edges
    
    TODO: Modify the epidata while adding a new edge"""


    edges= [(source, role, target)]#adding the new edge
    edges.extend(graph.edges())

    #modified amr after adding the required edge
    modified = penman.Graph(graph.instances() + edges + graph.attributes())
    return modified

def coreference_edges(merged_amr, name, amr_coref):

    """ Function to add coreference edges to the merged graph. The input is the combined AMR graph of all smaller
    comment AMR graphs."""
    d = amr_coref[name]
    for relation, cluster in d.items():
        var = [i[1] for i in cluster]
        #amr_coref is sorted according to time (i.e. comments appearing first temporally appear before) by default
        source = var[0] #the directed edge will start from the comment appearing first: following temporal fashion
        for target in var[1:]:#add :COREF edges from the source word to all words in the cluster
            added = add_edge(merged_amr, source, ":COREF", target)
            merged_amr = added
    return merged_amr

def concept_merge(modified):
    normalised_graph = normalise_graph(modified)
    word2var = generate_word2var(normalised_graph)
    for word, var in word2var.items():
        head_node = var[0]
        if len(var)>1:
            for j in var[1:]:
                added = add_edge(normalised_graph, head_node, ":SAME", j)
                normalised_graph = added
    return normalised_graph

def generate_word2var(normalised_graph):
    """This function returns a word-to-variableNames dictionary.
    
    A word might be present on multiple nodes. This returns the dictionary storing the nodes for every word."""
    word2var = {}# a dictionary mapping the words to nodes; example: the word 'name' might belong to 2 nodes

    for (source, role, target) in normalised_graph.instances():
        if target in word2var:
            word2var[target].append(source)
        else:
            word2var[target] = [source]
    return word2var

def normalise_graph(modified):

    """A function to convert the concepts in the amr to meaningful form so that we can apply glove embedding later on
    to find their node representations. 
    
    Note: Removing the part after the hyphen (-) in many of the concept names."""
    normalised_instances = []
    for (source, role, target) in modified.instances():
        if "-" in target:#for example: "save-01" concept is converted to "save". 
            normalised_instances.append((source, role, target[:target.rfind("-")]))
        else:
            normalised_instances.append((source, role, target))
    normalised_graph = penman.Graph(normalised_instances + modified.edges() + modified.attributes())
    return normalised_graph

def modify_variables(amr, i):

    """This function takes an AMR as input and modifies the variables of the AMR depending on the serial number
    of the AMR. Here, (i) refers to the (i)th comment on a particular news piece.
    
    Returns: The modifed penman Graph of the AMR string.
    
    Note: This function does not modify the edpidata or metadata of the input AMR. We just modify the variable names 
    in this function. Since our ultimate goal is to merge several AMR graphs, it is highly likely that different
    amrs have the same variable names. Thus to distinguish between variables of different amrs we assign unique names
    to different variables."""

    g = penman.decode(amr)

    g_meta = add_lemmas(amr, snt_key='snt')#adding lemmas , tokens to the AMR string
    
    #create a dictionary for mapping old variables to new variable names

    var, d = list(g.variables()), {}

    for j in range(len(var)):
        d[var[j]] = "c{}-{}".format(i, j)
        
    #modify the variable names of instances, edges, attributes of the original amr graph 
    instances, edges, attributes, epidata = [], [], [], {}
    for source, role, target in g.instances():#modify the instances
        instances.append((d[source], role, target))
    for source, role, target in g.edges():#modify the edges
        edges.append((d[source], role, d[target]))

    for source, role, target in g.attributes():#modify the attributes
        attributes.append((d[source], role, target))

    for (source, role, target) in g.epidata.keys():#modify the attributes
        
        push_pop = g.epidata[(source, role, target)]
        
        modified_epi = []
        for p in push_pop:
            if isinstance(p, penman.layout.Push):  modified_epi.append(penman.layout.Push(d[p.variable]))
            elif isinstance(p, penman.layout.Pop):  modified_epi.append(p)
            else: print(p)
    
        #if the epidata key is either an instance or attribute triple
        if (source, role, target) in g.instances() or (source, role, target) in g.attributes(): 
            epidata[(d[source], role, target)] = modified_epi
        
        elif (source, role, target) in g.edges(): 
            epidata[(d[source], role, d[target])] = modified_epi
        else:
            print((source, role, target))
        
    modified  = penman.Graph(instances + edges + attributes)#return the modifies graph 
    
    modified.metadata = g_meta.metadata #using the metadata from the original graph
    
    modified.epidata = epidata #using the epidata from the original graph -- name changed

    assert len(eval(modified.metadata['lemmas']))==len(eval(modified.metadata['tokens'])), "Length of tokens must be equal to lemmas"

    return modified

def modify_amr(datas_path, save_path=None):
    """
    input: list of dataframes dealed by tweets_to_amr function
    output: list of penman graphs
    """
    amr_data = []
    u_id = []
    user_ids = []
    labels = []
    ignored = []
    lv = 0
    for data_path in datas_path:
        filename_with_extension = os.path.basename(data_path)
        u_id.append(os.path.splitext(filename_with_extension)[0])
        
    for i in range(len(u_id)):
        print(u_id[i])
        q = datas_path[i]
        with open(datas_path[i], 'r', encoding='utf-8') as f:
            now_user = json.load(f)
        # modify all Comment AMRs belonging to one news article
        modified_amr_list = []
        if not now_user['amr']:
            continue
                
        try:
            if save_path:
                check_path = os.path.join(save_path)
                check_path = os.path.join(check_path, f"{u_id[i]}.amr.pennam")
                os.makedirs(os.path.dirname(check_path), exist_ok=True)
                try:
                    penman.load(check_path, model = NoOpModel())
                    continue
                except:
                    print(check_path, "error")
            for k in range(len(now_user['amr'])):
                var_mod = modify_variables(now_user['amr'][k], k+1)
                modified_amr_list.append(var_mod)
            user_ids.append(u_id[i])
            amr_data.append(modified_amr_list)
            if save_path:
                path = os.path.join(save_path)
                path = os.path.join(path, f"{u_id[i]}.amr.pennam")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    penman.dump(modified_amr_list, f, model = NoOpModel())
            modified_amr_list = penman.dumps(modified_amr_list, model=NoOpModel())
        except AssertionError:
            print("**************Ignoring the file {}******************".format(q))
            ignored.append(q)
        except:
            print("**************Ignoring the file decode error{}******************".format(q))
            ignored.append(q)
        lv+=1
        # break
        # print("Done", lv, len(modified_amr_list))
    print("Files ignored:\n{}".format(ignored))
    print("{} filed ignored".format(len(ignored)))
    return amr_data, user_ids

def merge_amr(amr_data, u_id, save_path=None):
    """
    input: list of penman graphs
    output: merged penman graph
    """
    merged_data = []
    lv=0
    for i in range(len(amr_data)):
        # modified_amr_list = penman.loads(amr_data[i], model = NoOpModel())
        modified_amr_list = amr_data[i]
        #adding dummy node and :COMMENT edges
        instances, edges, attributes = [('d', ':instance', 'dummy')], [], []
        metadata, epidata= {'snt': '', 'lemmas' : [], 'tokens' : []}, {('d', ':instance', 'dummy') : []}
        subgraph_list = []

        for graph in modified_amr_list:
            #for now we are ignoring the :coref and :same-as nodes to be a part of any subgraph - later on they can be added
            node_list = [source for source, _, _ in graph.instances()]#maintained for creating subgraphs later on
            subgraph_list.append(node_list)
            edges.append(('d', ':tweets', graph.top))
            instances.extend(graph.instances())
            edges.extend(graph.edges())
            attributes.extend(graph.attributes())
            metadata['snt']+= "{} ".format(graph.metadata['snt'])
            metadata['lemmas'].extend(ast.literal_eval(graph.metadata['lemmas']))
            metadata['tokens'].extend(ast.literal_eval(graph.metadata['tokens']))
            #adding epidata for the added edge
            epidata[('d', ':COMMENT', graph.top)] = [penman.layout.Push(graph.top)]
            #Adding the epidata from all amrs
            epidata.update(graph.epidata)
        metadata['tokens'] = json.dumps(metadata['tokens']) #convert metadata tokens to string format
        metadata['lemmas'] = json.dumps(metadata['lemmas'])
        #final modified graph consisting of all comment AMRs corresponding to one news piece
        modified = penman.Graph(instances + edges + attributes)
        modified.metadata = metadata
        modified.epidata = epidata
        #adding the coreference edges to the merged amr graph
        ##
        #adding the concept merging edges to the merged amr graph
        try:
            modified = concept_merge(modified)
        except:
            continue
        modified.metadata['subgraphs'] = json.dumps(subgraph_list)#adding the subgraph information
        #here every subgraph corresponds to a particular comment on the news article and the merged amr is the overall combination of all such subgraphs
        #storing the final AMR graph (merged + coreference + concept merging)
                    
        path = os.path.join(save_path, f'{u_id[i]}_merge_amr.amr.penman')
        with open(path, 'w') as f:
            penman.dump([modified], f, model = NoOpModel())
        merged_data.append(modified)
        print("Done", lv, u_id[i])
        lv+=1
    return merged_data

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    datasets = ["cresci-2015-data", "cresci-2017-data", "cresci-stock-2018", "midterm-2018", "twibot-20"]
    texts = ['description', 'tweets']
    features = ['sentiments', 'emotions', 'topics', 'values']


    for dataset in datasets:
        for text in texts:
            for feature in features:
                
                os.makedirs(save_path, exist_ok=True)
                dir_path = f'./8AMR_CIGA/data/amr_gen/{dataset}/{text}/{feature}'
                datas_path = glob.glob(dir_path + '/*.json')
                
                amr_data, user_ids = modify_amr(datas_path, save_path=save_path)
                print(len(user_ids),"/", len(datas_path))

                save_path = f'./8AMR_CIGA/data/amr_merge/{dataset}/{text}/{feature}'
                os.makedirs(save_path, exist_ok=True)
                merged_data = merge_amr(amr_data, user_ids, save_path)
                print(len(merged_data), '/', len(user_ids))

