This is some initial sample data to show how the code works.

The data here is for 1 ion solvated in TIP3P water at 21 sampled states

## Usage

Simply copy the contents of this folder to the same directory as esq\_construct\_ukln.py, rename "n21\_init.npy" to "qes.npy", then run 

```python
python esq_construct_ukln.py --nstates 21
```

This wlll run the free energy analysis, clustering, and movie creation for this data 
