import argparse
import subprocess
import time
import datetime
import os

semi_synth_datasets = [
    "CRIMES-D1",
    "DJ30-D1",
    "GAS-D1",
    "OLYMPIC-D1",
    "POKER-D1",
    "SENSOR-D1",
    "TAGS-D1",
    "ACTIVITY_RAW-D1",
    "ACTIVITY-D1",
    "CONNECT4-D1",
    "COVERTYPE-D1",
]

real_datasets = [
    "activity",
    "connect-4",
    "CovPokElec",
    "covtype",
    "crimes",
    "fars",
    "gas",
    "hypothyroid",
    "kddcup",
    "kr-vs-k",
    "lymph",
    "olympic",
    "poker",
    "sensor",
    "shuttle",
    "tags",
    "thyroid",
    "zoo",
]

algorithms = [
    "moa.classifiers.trees.GHVFDT",
    "moa.classifiers.trees.HDVFDT",
    "moa.classifiers.meta.imbalanced.ROSE",
    "moa.classifiers.meta.OzaBagAdwin",
    "moa.classifiers.meta.LeveragingBag",
    "moa.classifiers.meta.AdaptiveRandomForest",
    "moa.classifiers.meta.OOB",
    "moa.classifiers.meta.UOB",
]

al_strategies = ["Random"]


def cmdlineparse(args):
    parser = argparse.ArgumentParser(description="Run MOA scripts")

    parser.add_argument(
        "--datasets", type=str, default=None,
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="results/uncertainty-kappaoversampling/",
    )

    parser.add_argument(
        "--max-processes", type=int, default=4,
    )

    args = parser.parse_args(args)
    return args


def train(args):

    if args.datasets == "SEMI":
        datasets = semi_synth_datasets
        args.results_path = args.results_path + "semi-synth/"
    else:
        datasets = real_datasets
        args.results_path = args.results_path + "real/"

    VMargs = "-Xms8g -Xmx1024g"
    jarFile = "kappaoversampling-1.0-jar-with-dependencies.jar"

    al_budget = ["1.0", "0.5", "0.2", "0.1", "0.01", "0.05", "0.005", "0.001"]

    results = [
        (dt, strategy, budget, alg)
        for dt in datasets
        for strategy in al_strategies
        for budget in al_budget
        for alg in algorithms
    ]

    print(
        f'>>>>>> START: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} >>>>>>'
    )

    processes = list()
    success_count = 0

    for (dataset, strategy, budget, alg) in results:
        if strategy == "Random":
            cmd = (
                "java "
                + VMargs
                + " -javaagent:sizeofag-1.0.4.jar -cp "
                + jarFile
                + " "
                + "moa.DoTask moa.tasks.meta.ALPrequentialEvaluationTask"
                + ' -e "(ALMultiClassImbalancedPerformanceEvaluator -w 500)"'
                + ' -s "(ArffFileStream -f datasets/real/'
                + dataset
                + '.arff)"'
                + ' -l "(moa.classifiers.active.ALRandom -l (moa.classifiers.meta.imbalanced.KappaOversampling -l '
                + alg
                + ") -b (moa.classifiers.active.budget.FixedBM -b "
                + budget
                + '))"'
                + " -f 500"
                + " -d "
                + args.results_path
                + alg.split(".")[-1]
                + "-"
                + dataset
                + "-"
                + strategy
                + "-"
                + budget
                + ".csv"
            )
        else:
            cmd = (
                "java "
                + VMargs
                + " -javaagent:sizeofag-1.0.4.jar -cp "
                + jarFile
                + " "
                + "moa.DoTask moa.tasks.meta.ALPrequentialEvaluationTask"
                + ' -e "(ALMultiClassImbalancedPerformanceEvaluator -w 500)"'
                + ' -s "(ArffFileStream -f datasets/real/'
                + dataset
                + '.arff)"'
                + ' -l "(moa.classifiers.active.ALUncertainty -l (moa.classifiers.meta.imbalanced.KappaOversampling -l '
                + algorithms[alg]
                + ") -d "
                + strategy
                + " -b "
                + budget
                + ')"'
                + " -f 500"
                + " -d "
                + args.results_path
                + alg.split(".")[-1]
                + "-"
                + dataset
                + "-"
                + strategy
                + "-"
                + budget
                + ".csv"
            )

        print(cmd)
        processes.append(subprocess.Popen(cmd, stdout=subprocess.PIPE))

        while len(processes) >= args.max_processes:
            time.sleep(30)

            # Print outputs and remove finished processes from list
            finished_processes = [
                i for i, p in enumerate(processes) if p.poll() is not None
            ]
            for finished_i in finished_processes:
                try:
                    comm = processes[finished_i].communicate()
                    print(comm[0].decode("utf-8"))
                    return_code = processes[finished_i].poll()
                    if return_code == 0:
                        success_count += 1
                except:
                    pass
                processes = [
                    p for i, p in enumerate(processes) if i != finished_i
                ]

    while len(processes):
        time.sleep(30)

        # Print outputs and remove finished processes from list
        finished_processes = [
            i for i, p in enumerate(processes) if p.poll() is not None
        ]
        for finished_i in finished_processes:
            try:
                comm = processes[finished_i].communicate()
                print(comm[0].decode("utf-8"))
                return_code = processes[finished_i].poll()
                if return_code == 0:
                    success_count += 1
            except:
                pass
            processes = [p for i, p in enumerate(processes) if i != finished_i]

    print(
        f'>>>>>> END train_all_asins: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} >>>>>>'
    )


def main(args=None):
    args = cmdlineparse(args)
    train(args)


if __name__ == "__main__":
    main()
