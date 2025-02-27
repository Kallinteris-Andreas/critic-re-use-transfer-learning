#python plot.py --result_directory models/MATD3-cc_2x4_TNL models/MATD3-cc_2x4d_Ant_TNL models/MATD3-cc_4x2_Ant_TNL


# plot all
python plot.py --result_directory results/vault/TD3_Ant-v5_1697532089.9189715/ results/vault/MATD3_2x4_Ant_1700507148.252164/ results/vault/MATD3_2x4d_Ant_1700507326.8253415/ results/vault/MATD3_4x2_Ant_1700480193.70154/ results/vault/MATD3-cc_2x4_Ant_1700480061.9332101/ results/vault/MATD3-cc_2x4d_Ant_1700480106.8648412/ results/vault/MATD3-cc_4x2_Ant_1700480166.2900493/  --mode average
python plot.py --result_directory results/vault/TD3_Ant-v5_1697532089.9189715/ results/vault/MATD3_2x4_Ant_1700507148.252164/ results/vault/MATD3_2x4d_Ant_1700507326.8253415/ results/vault/MATD3_4x2_Ant_1700480193.70154/ results/vault/MATD3-cc_2x4_Ant_1700480061.9332101/ results/vault/MATD3-cc_2x4d_Ant_1700480106.8648412/ results/vault/MATD3-cc_4x2_Ant_1700480166.2900493/  --mode max


# plot just basic learning (no Trasfer learning)
#python plot.py --result_directory results/vault/TD3_Ant-v5_1697532089.9189715/ results/vault/MATD3_2x4_Ant_1700507148.252164/ results/vault/MATD3_2x4d_Ant_1700507326.8253415/ results/vault/MATD3_4x2_Ant_1700480193.70154/ results/vault/MATD3-cc_2x4_Ant_1700480061.9332101/ results/vault/MATD3-cc_2x4d_Ant_1700480106.8648412/ results/vault/MATD3-cc_4x2_Ant_1700480166.2900493/  --mode average
#python plot.py --result_directory results/vault/TD3_Ant-v5_1697532089.9189715/ results/vault/MATD3_2x4_Ant_1700507148.252164/ results/vault/MATD3_2x4d_Ant_1700507326.8253415/ results/vault/MATD3_4x2_Ant_1700480193.70154/ results/vault/MATD3-cc_2x4_Ant_1700480061.9332101/ results/vault/MATD3-cc_2x4d_Ant_1700480106.8648412/ results/vault/MATD3-cc_4x2_Ant_1700480166.2900493/  --mode max
