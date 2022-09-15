#pragma once

namespace extremeopt{
    struct Parameters
    {
        int max_iters = 500;
        bool do_newton = false;
        int ls_iters = 200;
        double E_target = 10.0;
        /* data */
    };
    
} // namespace extremeopt