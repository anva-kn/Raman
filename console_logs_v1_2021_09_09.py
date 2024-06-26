from tools.ramanflow.read_data import ReadData as RD
_, check = RD.read_dir_tiff_files('data/20210319 MG colloidal SERS test')
f_sup_0319, MG_colloidal_SERS_test_20210319 = RD.read_dir_tiff_files('data/20210319 MG colloidal SERS test')
f_sup_0331, MG_colloidal_SERS_test_20210331 = RD.read_dir_tiff_files('data/20210331 MG colloidal SERS2')
_, batch2_power_test_20210331 = RD.read_dir_tiff_files('data/20210331 MG colloidal SERS2/batch2_power_test')
_, HWP36_to_80_step2_10s_20210331 = RD.read_dir_tiff_files('data/20210331 MG colloidal SERS2/batch2_power_test/10s_from_HWP36_to_80_step2')
f_sup_0427, MG_ACE_ACETAMIPRID_colloidal_SERS3_20210427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3')
_, MG_1_5ppb_0427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/1_5ppb_MG')
_, MG_15ppb_0427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/15ppb_MG')
_, MG_150ppb_0427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG')
_, ACE_0427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/ACE')
_, Acetamiprid_0427 = RD.read_dir_tiff_files('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/Acetamiprid')
f_sup_0707, Carbendazim_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim')
_, Carbendazim_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom')
_, Thiacloprid_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid')
_, Thiacloprid_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom')
_, Imidacloprid_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid')
_, Imidacloprid_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Acetamiprid_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid')
_, Acetamiprid_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Acephate_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/5_Acephate')
_, Acephate_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, MG_150ppb_0707 = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/150ppb_MG test')
_, MG_150ppb_0707_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/150ppb_MG test/HWP42 to 82 step4 1s 5x5map 10xzoom new colloids')
_, Acetamiprid_with_150ppb_MG = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/Mix Acetamiprid with 0.1ml of 150ppb_MG/2ml Acetamiprid colloidal plus 0.1ml 150ppb_MG mix 1s 32x32 HWP50.tif')
_, Acetamiprid_with_150ppb_MG_power_test = RD.read_dir_tiff_files('data/20210707 colloidal SERS multiple analytes/Mix Acetamiprid with 0.1ml of 150ppb_MG/HWP42 to 62 step2 1s 6x6map 10xzoom')
f_sup_0722, colloidal_SERS_multiple_analytes_20210722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes')
_, Carbendazim_0722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/1_Carbendanzim')
_, Carbendazim_0722_power_test = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Thiacloprid_0722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/2_Thiacloprid')
_, Thiacloprid_0722_power_test = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Imidacloprid_0722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/3_Imidacloprid')
_, Imidacloprid_0722_power_test = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Acetamiprid_0722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/4_Acetamiprid')
_, Acetamiprid_0722_power_test = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom')
_, Acephate_0722 = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/5_Acephate')
_, Acephate_0722_power_test = RD.read_dir_tiff_files('data/20210722 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom')

