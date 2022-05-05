from operator import index
from textwrap import wrap
import tkinter as tk
from tkinter import *
from data.loader import DataLoader, parse_file
from program_synthesis.heuristic_generator import HeuristicGenerator
import numpy as np
import pickle as pkl
from clustering import generate_clusters

'''
All selection methods need the following type signature
index to points will be of the same type as the self.index_to_points object in the UI class.
'''


def random_batch_select(index_to_points, batch_size, iteration):
    selected_vals = np.random.choice(list(index_to_points.keys()), size=batch_size, replace=False)
    return [(key, index_to_points[key]) for key in selected_vals]


def active_sampling(index_to_points, batch_size, iteration):
    selected_vals = np.arange(iteration*batch_size, (iteration+1)*batch_size)
    return [(key, index_to_points[key]) for key in selected_vals]


class SnubaBaseUI:

    def __init__(self, path_to_dataset="./data/imdb/budgetandactors.txt", path_to_embeddings="./data/imdb/corpus_embedding.pickle", batch_size=5, selection_method=active_sampling, minimum_points=50,
                 outfilepath="labeled_datapoints"):
        self.gui = tk.Tk()
        self.gui.geometry("1200x700")
        #load dataset
        self.batch_size = batch_size
        self.selection_method = random_batch_select if selection_method is None else selection_method
        plots, _ = parse_file(path_to_dataset)
        if self.selection_method == random_batch_select:
            self.index_to_points = dict(enumerate(plots))
        else:
            cluster_data, self.cluster_choice = generate_clusters(path_to_embeddings, plots)
            self.index_to_points = dict(enumerate(cluster_data))

        self.labeled_points = []
        self.total_num_datapoints = len(plots)
        self.minimum_required_points = minimum_points
        self.outfilepath = outfilepath
        self.iteration = 0
        self.num_clusters = 50
        
        self.initialize_ui_fields()
        self.update_metrics()

        self.display_data_batch()

        self.gui.mainloop()

    def initialize_ui_fields(self):
        # build out the UI using tkinter
        self.current_frame = tk.Frame(self.gui, width=600, height=700)
        self.current_frame.pack(side=tk.LEFT)
        label = Label(self.current_frame, text="Data Points",
                      font=('Helvetica', 18, 'bold')).pack()
        self.labeled_frame = tk.Frame(self.gui, width=600, height=700)
        self.labeled_frame.pack(side=tk.RIGHT)

        self.current_datapoints_frame = tk.Label(
            self.current_frame, wraplength=500, justify=tk.LEFT)
        self.current_datapoints_frame.pack(side=tk.LEFT)
        self.current_datapoints_frame.pack(side=tk.TOP)
        self.answer_frames = []
        self.radiobutton_pairs = []
        self.vars_list = []
        for idx in range(self.batch_size):
            frame = tk.Frame(self.current_frame)
            frame.pack(side=tk.TOP)
            tk.Label(frame, text="Example {}".format(
                idx + 1)).pack(side="left")
            var = tk.IntVar(value=0)
            self.vars_list.append(var)
            rb1 = tk.Radiobutton(frame, text="Action", variable=var, value=1)
            rb1.pack(side="left")
            rb2 = tk.Radiobutton(frame, text="Romance", variable=var, value=-1)
            rb2.pack(side="left")
            self.answer_frames.append(frame)
            self.radiobutton_pairs.append((rb1, rb2))
        self.labeled_datapoints_frame = tk.Listbox(
            self.labeled_frame, width=65, height=35)
        self.labeled_datapoints_frame.pack(side=tk.TOP, anchor=tk.NW)

        self.labeled_datapoints_scrollbar = tk.Scrollbar(self.labeled_frame)
        self.labeled_datapoints_scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.labeled_datapoints_frame.config(
            yscrollcommand=self.labeled_datapoints_scrollbar.set)
        self.labeled_datapoints_scrollbar.config(
            command=self.labeled_datapoints_frame.yview)
        self.finish_button = tk.Button(
            self.labeled_frame, text="Finish!", state=tk.DISABLED, command=self.finish_trigger)
        self.finish_button.pack(side=tk.BOTTOM)
        self.current_buttons = tk.Frame(self.current_frame)
        self.current_buttons.pack(side=tk.BOTTOM)
        self.new_batch_button = tk.Button(
            self.current_buttons, text="Fetch new example batch", command=self.display_data_batch)
        self.new_batch_button.pack(side=tk.LEFT)
        self.submit_current_button = tk.Button(
            self.current_buttons, text="Submit current label batch", command=self.submit_trigger)
        self.submit_current_button.pack(side=tk.RIGHT)
        self.metrics_frame = tk.Label(self.labeled_frame)
        self.metrics_frame.pack(side=tk.BOTTOM)

    def submit_trigger(self):
        # add the new examples, update the metrics, and display a new batch
        self.add_and_display_labeled_indexes()
        self.update_metrics()
        if len(self.labeled_points) >= self.minimum_required_points:
            self.finish_button['state'] = tk.NORMAL
        for idx in range(len(self.vars_list)):
            self.vars_list[idx].set(0)
            self.radiobutton_pairs[idx][0].deselect()
            self.radiobutton_pairs[idx][1].deselect()
        self.display_data_batch()

    def finish_trigger(self):
        # write out the collected examples to a file, and close
        with open(self.outfilepath, "wb") as fle:
            pkl.dump(self.labeled_points, fle)
        self.gui.destroy()

    def update_metrics(self):
        # update the # of currently labeled out of the total
        self.metrics_string = ""
        self.current_ratio = "Number of data points labeled so far (out of total): {}/{}\n".format(
            len(self.labeled_points), self.total_num_datapoints)
        
        self.metrics_string += self.current_ratio
        if self.selection_method == active_sampling:
            labeled_indexes = [l[0] for l in self.labeled_points]
            cluster_coverage = [self.cluster_choice[i] for i in labeled_indexes]
            cluster_coverage = list(set(cluster_coverage))
            self.coverage = "Percentage of clusters covered so far: {}%\n".format((len(cluster_coverage)/self.num_clusters)*100)
            self.metrics_string += self.coverage
        # if there are additional metrics, include them here and add them to self.metrics_string.
        self.metrics_frame["text"] = self.metrics_string


    def add_and_display_labeled_indexes(self):
        for curidx, (idx, plot) in enumerate(self.current_index_data_pairs):
            current_var = self.vars_list[curidx]
            if current_var.get() == 0:
                #unlabeled, don't count it
                continue
            label = "ROMANCE" if current_var.get() == -1 else "ACTION"
            if idx in self.index_to_points:
                _ = self.index_to_points.pop(idx)
                self.labeled_points.append((idx, plot, current_var.get()))
                # update and display these in the right region
                self.labeled_datapoints_frame.insert(
                    tk.END, "{}. {} \n LABEL: {}".format(idx, plot, label))

    def display_data_batch(self):
        self.current_index_data_pairs = self.selection_method(
            self.index_to_points, self.batch_size, self.iteration)
        self.iteration += 1
        display_string = ""
        # display the selected datapoints
        cntr = 1
        for (idx, plot) in self.current_index_data_pairs:
            display_string += "{}. {} \n\n".format(cntr, plot)
            cntr += 1
        self.current_datapoints_frame.configure(text=display_string)


ui = SnubaBaseUI()
