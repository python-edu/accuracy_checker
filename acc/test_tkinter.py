import tkinter as tk
from tkinter import filedialog
from tkinter import ttk




def get_file(key: int):
    file_no = {1: {'type_name': "CSV / Raster",
                   'filters': ("*.csv", "*.tif", "*.tiff", "*.TIF", "*.TIFF")
                   },
              2: {'type_name': "Raster / vector",
                   'filters': ("*.tif", "*.tiff", "*.TIF", "*.TIFF",
                              "*.shp", "*.gpkg")
                   },
              3: {'type_name': "JSON",
                   'filters': ("*.json")
                   }
               }

    type_name = file_no[key]['type_name']
    filters = file_no[key]['filters']

    file = filedialog.askopenfilename(
            title="File",
            filetypes=[(type_name, filters)])
    return file


if __name__ == "__main__":
    root = tk.Tk()

    b1 = tk.Button(root, text='Quit', command=root.destroy)
    no = 1
    b2 = ttk.Button(root, text='File', command=lambda no=no: get_file(no))
    b1.pack()
    b2.pack()
    # filedialog.Directory()
    tk.filedialog.askdirectory()
    root.mainloop()
