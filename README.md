# Hypertension-Risk-Modeling-Analysis
A complete pipeline for analyzing NHANES data to model hypertension risk using logistic regression.

    To run this analysis:
    
    1. Download NHANES 2017-2018 datasets from:
       https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
       
    2. Required files:
       - DEMO_J.XPT (Demographics)
       - BPX_J.XPT (Blood Pressure & Cholesterol)
       - BMX_J.XPT (Body Measures)
    
    3. Update file paths in the code
    
    4. Run: results = analysis.run_complete_analysis(demo_path, bpx_path, bmx_path)
    
    The analysis will generate:
    - Comprehensive EDA plots
    - Model comparison results
    - Statistical diagnostics
    - Decision matrix
    - Excel export with all results
