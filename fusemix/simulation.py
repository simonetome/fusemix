"""
Simulation runs the pipeline for a grid of

MD configurations
Datasets


form of md param grid

{
'props' = []
'mf_proportions' = []
'mnar_proportions' = []

}

"""

from fusemix.data_loading import load_uciml
from fusemix.pipeline import Pipeline


class Simulation:
    def __init__(self, id_datasets, n_runs, md_param_grid):
        self.id_datasets = id_datasets
        self.n_runs = n_runs
        self.md_param_grid = md_param_grid

        self.all_datasets = {}

        self.complete_vs_true = {}
        self.fuse_vs_true = {}
        self.fuse_vs_complete = {}
        self.knn_vs_true = {}
        self.knn_vs_complete = {}

    def __load_datasets(self):
        for _id in self.id_datasets:
            self.all_datasets[_id] = load_uciml(_id)

    def __run_pipeline(self, data, prop, mf_proportion, mnar_proportion, seed):
        """
        run n_runs for a single configuration {dataset, md_params}
        """
        # Set parameters for pipeline
        pipeline_params = {}
        pipeline_params['seed'] = seed
        pipeline_params['prop'] = prop
        pipeline_params['mnar_freq'] = mnar_proportion
        pipeline_params['mf_proportion'] = mf_proportion

        # Fixed parameters
        pipeline_params['mice_iterations'] = 2
        pipeline_params['num_imputations'] = 10
        pipeline_params['nn_snf'] = 10

        pipeline_params['complete_data'] = data['X']
        pipeline_params['cat_mask'] = data['cat_mask']
        pipeline_params['y_data'] = data['y']

        pipeline = Pipeline(pipeline_params=pipeline_params,verbose = False)
        pipeline.run()
        return pipeline

    def run(self):
        """

        Returns:

        """
        print("Loading datasets")
        self.__load_datasets()

        for d in self.all_datasets.keys():
            data = self.all_datasets[d]
            for prop in self.md_param_grid['props']:
                for mf_proportion in self.md_param_grid['mf_proportions']:
                    for mnar_proportion in self.md_param_grid['mnar_proportions']:

                        key = (d,prop,mf_proportion,mnar_proportion)
                        self.complete_vs_true[key] = []
                        self.fuse_vs_true[key] = []
                        self.fuse_vs_complete[key] = []
                        self.knn_vs_true[key] = []
                        self.knn_vs_complete[key] = []

                        for s in range(self.n_runs):
                            try:
                                print("Running simulation run #{}".format(s))
                                pipeline = self.__run_pipeline(data=data,
                                                               prop=prop,
                                                               mf_proportion=mf_proportion,
                                                               mnar_proportion=mnar_proportion,
                                                               seed=s)

                                tests = ['complete_vs_true',
                                            'fuse_vs_true',
                                            'fuse_vs_complete',
                                            'knn_vs_true',
                                            'knn_vs_complete']

                                for test in tests:
                                    result = getattr(pipeline, test)
                                    getattr(self, test)[key].append({
                                        'seed': s,
                                        'ARI': result['ARI'],
                                        'AMI': result['AMI']
                                    })

                            except Exception as e:
                                print(f"An error occurred: {e}")