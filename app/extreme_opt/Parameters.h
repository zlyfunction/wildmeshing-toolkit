#pragma once
#include <string>
namespace extremeopt {
struct Parameters
{
    std::string model_name;
    bool save_meshes = false;

    int max_iters = 500;
    bool do_newton = false;
    bool do_swap = false;
    bool do_collapse = false;
    bool do_split = false;
    bool local_smooth = false;
    bool global_smooth = true;
    bool use_envelope = false;
    int global_upsample = 0;
    int ls_iters = 200;
    double E_target = 10.0;
    double elen_alpha = 2.0;
    bool with_cons = true;
    bool do_projection = true;

    int split_succ_cnt = 300;
    bool use_max_energy = false;
    int Lp = 4;
    /* data */
};

} // namespace extremeopt
