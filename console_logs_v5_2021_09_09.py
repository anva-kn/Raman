from tools.ramanflow.read_data import ReadData as RD
import numpy as np

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

f_sup_0810, Carbendazim_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1')
_, Carbendazim_Acetamiprid_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4')
_, Carbendazim_Acetamiprid_Acephate_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4+5')
_, Carbendazim_Acephate_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+5')
_, Acetamiprid_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4')
_, Acetamiprid_Acephate_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4+5')
_, Acephate_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/5')
_, colloids_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/colloidal solution')

f_sup_0811, Carbendazim_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1')
_, Carbendazim_Acetamiprid_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4')
_, Carbendazim_Acetamiprid_Acephate_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4+5')
_, Carbendazim_Acephate_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+5')
_, Acetamiprid_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4')
_, Acetamiprid_Acephate_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4+5')
_, Acephate_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/5')
_, MG_150ppb_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/MG_150ppb')

f_sup_0813, Carbendazim_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1')
_, Carbendazim_Acetamiprid_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4')
_, Carbendazim_Acetamiprid_Acephate_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4+5')
_, Carbendazim_Acephate_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+5')
_, Old_Carbendazim_old_colloids_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1o+o')
_, Carbendazim_old_colloids_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+o')
_, Acetamiprid_Acephate_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+5')
_, Acetamiprid_old_colloids_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+o')
_, Old_Acetamiprid_old_colloids_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4o+o')
_, Acetamiprid_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4')
_, Acephate_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/5')

f_sup_0818, Carbendazim_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1')
_, Carbendazim_Acetamiprid_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4')
_, Carbendazim_Acetamiprid_Acephate_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4+5')
_, Carbendazim_Acephate_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+5')
_, Carbendazim_old_colloids_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+o')
_, Carbendazim_old_colloids_v2_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1o+Co')
_, Carbendazim_old_colloids_v3_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1oo+oo')
_, Acetamiprid_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4')
_, Acetamiprid_Acephate_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+5')
_, Acetamiprid_old_colloids_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+o')
_, Acetamiprid_old_colloids_v2_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4o+o')
_, Acetamiprid_old_colloids_v3_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4oo+oo')
_, Acephate_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/5')

f_sup_unknown, analyte_unknown = RD.read_dat_file('data/anvar 10-29/area1_Exported.dat')

f_sup_set_1, MG_set1 = RD.read_dir_tiff_files('data/MG set 1')

f_sup_set_3, MG_set3 = RD.read_dir_dat_files('data/MG-3_15000_ppb')

f_sup_set_4, MG_set4 = RD.read_dir_dat_files('data/MG-4_1500_ppb')

f_sup_set_5, MG_set5 = RD.read_dir_dat_files('data/MG-5_150_ppb')

f_sup_mix_analytes, Acephate_10_2M = RD.read_dir_tiff_files('data/SERS with mix analyte/Acephate 10-2M 5715minutes')
_, MG_10_5M = RD.read_dir_tiff_files('data/SERS with mix analyte/MG 10-5M 930minutes')
_, mix_1115_min_1 = RD.read_tiff_file('data/SERS with mix analyte/mix/1115min/Mix-SERS-A-(1)-1115min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C).tif')
_, mix_1115_min_2 = RD.read_tiff_file('data/SERS with mix analyte/mix/1115min/Mix-SERS-A-(2)-1115min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C).tif')
mix_1115_min = np.concatenate((mix_1115_min_1, mix_1115_min_2), axis=0)

_, mix_4027_min_1 = RD.read_tiff_file('data/SERS with mix analyte/mix/4027min/Mix-SERS-A-(1)-4027min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C).tif')
_, mix_4027_min_2 = RD.read_tiff_file('data/SERS with mix analyte/mix/4027min/Mix-SERS-A-(2)-4027min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C).tif')
mix_4027_min = np.concatenate((mix_4027_min_1, mix_4027_min_2), axis=0)

f_sup_acephate, Acephate_exported = RD.read_dat_file('data/Acephate_Exported.dat')

MG = {**MG_10_5M, **MG_colloidal_SERS_test_20210319, **MG_colloidal_SERS_test_20210331, **MG_ACE_ACETAMIPRID_colloidal_SERS3_20210427,
      **MG_set5, **MG_set1, **MG_set3, **MG_set4, **MG_1_5ppb_0427, **MG_15ppb_0427, **MG_150ppb_0427, **MG_150ppb_0707,
      **MG_150ppb_0707, **MG_150ppb_0811, **MG_150ppb_0707_power_test, **MG_150ppb_0707_power_test, **HWP36_to_80_step2_10s_20210331,
      **batch2_power_test_20210331}

Carbendazim = {**Carbendazim_0707, **Carbendazim_0707_power_test, **Carbendazim_0722, **Carbendazim_0722_power_test,
               **Carbendazim_0810, **Carbendazim_0811, **Carbendazim_0813, **Carbendazim_0818, **Carbendazim_old_colloids_0813,
               **Carbendazim_old_colloids_0818, **Old_Carbendazim_old_colloids_0813}

Thiacloprid = {**Thiacloprid_0707, **Thiacloprid_0707_power_test, **Thiacloprid_0722, **Thiacloprid_0722_power_test}

Imidacloprid = {**Imidacloprid_0707, **Imidacloprid_0707_power_test, **Imidacloprid_0722, **Imidacloprid_0722_power_test}

Acetamiprid = {**Acetamiprid_0707, **Acetamiprid_0707_power_test, **Acetamiprid_0722, **Acetamiprid_0722_power_test,
               **Acetamiprid_0810, **Acetamiprid_0811, **Acetamiprid_0813, **Acetamiprid_0818, **Acetamiprid_0427,
               **Acetamiprid_old_colloids_0813, **Acetamiprid_old_colloids_0818, **Acetamiprid_old_colloids_v2_0818,
               **Acetamiprid_old_colloids_v3_0818, **Old_Acetamiprid_old_colloids_0813}

Acephate = {**Acephate_0707, **Acephate_0707_power_test, **Acephate_0722, **Acephate_0722_power_test, **Acephate_0810,
            **Acephate_0811, **Acephate_0813, **Acephate_0818, **Acephate_10_2M, **Acephate_exported, **ACE_0427}

Mixes = {**mix_1115_min, **mix_1115_min, **Carbendazim_Acephate_0810, **Carbendazim_Acephate_0811, **Carbendazim_Acephate_0813,
         **Carbendazim_Acephate_0818, **Acetamiprid_Acephate_0810, **Acetamiprid_Acephate_0813, **Acetamiprid_Acephate_0811,
         **Acetamiprid_Acephate_0818, **Carbendazim_Acetamiprid_Acephate_0810, **Carbendazim_Acetamiprid_Acephate_0811,
         **Carbendazim_Acetamiprid_Acephate_0813, **Carbendazim_Acetamiprid_Acephate_0818, **MG_ACE_ACETAMIPRID_colloidal_SERS3_20210427,
         }
