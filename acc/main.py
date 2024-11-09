import sys
import numpy as np
import pandas as pd
from pathlib import Path
from acc.src.args import parsuj_argumenty, info
from acc.src import data_recognition
from acc.src import functions as fn
from acc.src.args_data import args_func as afn
from acc.src.report import AccuracyReport


def main():
    parser = parsuj_argumenty()

    # 1. Obsługa argumentów linii poleceń
    # =====================================================================
    args = parser.parse_args()
    args = afn.args_validation(args,
                               **{'script_name': __file__, 'info': info})

    # --- scans data and checks data type ---
    args = data_recognition.specify_data_type(args)

    vb = fn.Verbose(args.verbose)
    vb(args, "Script arguments:", args_data=True)


    # 2. Odczyt danych
    # =====================================================================
    if hasattr(args, 'func'):
        read_data = args.func
    else:
        print(args)
        sys.exit()

    data, cross, cross_full, bin_cross, bin_cross_rep = read_data(args)
    vb(bin_cross, 'Binary confusion matrix:')


    # 3. Tradycyjne, klasyczne wskaźniki dokładności
    # =====================================================================
    classic_acc = fn.acc_from_cross(data, args)

    # Calculation results are displayed by default: verbose = True
    vb.verbose = True
    vb(classic_acc, 'Accuracy metrics classically used in remote sensing:')


    # 4. Nowe wskaźniki dokładności
    # =====================================================================
    simple_acc, complex_acc = fn.acc_from_bin_cross(bin_cross, args)
    vb(simple_acc, "Simple machine learning metrics:")
    vb(complex_acc, "Complex machine learning metrics:")

    # average
    m1 = np.round(simple_acc.mean(), 4)
    m2 = np.round(complex_acc.mean(), 4)

    average_acc = pd.DataFrame(pd.concat([m1, m2]))
    average_acc.columns = ["Value"]
    vb(average_acc, "Average machine learning metrics:")


    # 5.Saving data to files:
    #  - you can choose whether to save '*.csv' files or '*.zip' archives
    #    to disk (you can't save both)
    # =====================================================================

    vb.verbose = True
    data =  [("cross_full", cross_full),
             ("binary_cross", bin_cross),
             ("classic_acc", classic_acc),
             ("simple_acc", simple_acc),
             ("complex_acc", complex_acc),
             ("average_acc", average_acc)]
    df_dict = dict(data)
    df_dict = {key: val for key, val in df_dict.items() if val is not None}
    
    # if args.save and not args.zip:
    if hasattr(args, 'save') and not hasattr(args, 'zip'):
        recorded = fn.save_results(args.out_dir, df_dict)

        # Calculation results are displayed by default: verbose = True
        vb(recorded, "Results saved to `csv` files:")

    # elif args.zip and not args.save:
    elif hasattr(args, 'zip') and not hasattr(args, 'save'):
        fn.zip_results(args.zip_path, df_dict)
        
        vb(args.zip_path, "Results saved to zip archive:")


    # 6. Creates html report
    # =====================================================================
    # if args.report:
    if hasattr(args, 'report'):
        df_dict['bin_cross'] = bin_cross_rep

        titles = fn.format_title(['Confusion matrix',
                  'Binary confusion matrix',
                  'Remote sensing: classical metrics to evaluate image \
                          classification accuracy.',
                  'Machine learning: `simple` metrics for assessing image \
                          classification accuracy.',
                  'Machine learning: `composite` metrics for assessing image \
                          classification accuracy.',
                  'Machine Learning: Metrics Averages'])

        report_data = args.report_data.copy()
        report_data.update({'script_name': args.script_name})
        report =  AccuracyReport(**report_data)
        # breakpoint()
        data_dict = dict(zip(titles, df_dict.values()))
        res = report(data_dict)
        # breakpoint()
        with open(args.report_data['report_file'], 'w') as f:
            f.write(res)

        vb(args.report_data['report_file'], "An html report was generated:")


# ---

if __name__ == "__main__":
    main()


