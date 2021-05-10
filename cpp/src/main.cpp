#include "all.h"
#include <Eigen/Core>
#include "node.h"
#include "material.h"
#include "element_style.h"
#include "element.h"
#include "fem.h"
#include "io_data.h"
#include "input_wave.h"

using EV = Eigen::VectorXd ;
using EM = Eigen::MatrixXd ;

int main() {

  clock_t start = clock();

  // ----- Input FEM Mesh ----- //
  Fem fem = io_data::input_mesh("input/mesh.in");
  auto outputs = io_data::input_outputs("input/output.in");
  std::string output_dir = "result/";

  // ----- FEM Set up ----- //
  fem.set_init();
  fem.set_output(outputs);

  // ----- Define input wave ----- //
  size_t fsamp = 4000;
  double fp = 1.0;
  double duration = 9.0/fp;

  EV wave_acc;
  auto [tim, dt] = input_wave::linspace(0,duration,(int)(fsamp*duration));
  // wave_acc = input_wave::simple_sin(tim,fp,1.0);
  wave_acc = input_wave::ricker(tim,fp,1.0/fp,1.0);
  size_t ntim = tim.size();

  // std::ofstream f0(output_dir + "input.acc");
  // for (size_t it = 0 ; it < ntim ; it++) {
  //   f0 << tim(it) ;
  //   f0 << " " << wave_acc(it) ;
  //   f0 << "\n";
  // }
  // f0.close();
  // exit(1);

  // ----- Prepare time solver ----- //
  fem.update_init(dt);

  EM output_dispx = EM::Zero(1,fem.output_nnode);
  EM output_dispz = EM::Zero(1,fem.output_nnode);
  // EM output_velx = EM::Zero(1,fem.output_nnode);
  // EM output_velz = EM::Zero(1,fem.output_nnode);
  EM output_strxx = EM::Zero(1,fem.output_nelem);
  EM output_strzz = EM::Zero(1,fem.output_nelem);
  EM output_strxz = EM::Zero(1,fem.output_nelem);

  // ----- time iteration ----- //
  std::ofstream f_dispx(output_dir + "x.disp");
  std::ofstream f_dispz(output_dir + "z.disp");
  std::ofstream f_strxx(output_dir + "xx.str");
  std::ofstream f_strzz(output_dir + "zz.str");
  std::ofstream f_strxz(output_dir + "xz.str");
  
  EV acc0 = EV::Zero(fem.dof);
  EV vel0 = EV::Zero(fem.dof);

  for (size_t it = 0 ; it < ntim ; it++) {
    acc0[0] = wave_acc[it];
    vel0[0] += wave_acc[it]*dt;

    // fem.update_time_input_MD(vel0);
    fem.update_time_input_FD(vel0);

    for (size_t i = 0 ; i < fem.output_nnode ; i++) {
      Node* node_p = fem.output_nodes_p[i];
      output_dispx(0,i) = node_p->u(0);
      output_dispz(0,i) = node_p->u(1);
      // output_velx(0,i) = node_p->v(0);
      // output_velz(0,i) = node_p->v(1);
    }

    for (size_t i = 0 ; i < fem.output_nelem ; i++) {
      Element* element_p = fem.output_elements_p[i];
      output_strxx(0,i) = element_p->strain(0);
      output_strzz(0,i) = element_p->strain(1);
      output_strxz(0,i) = element_p->strain(2);
    }

    if (it%500 == 0) {
      std::cout << it << " t= " << it*dt << " ";
      std::cout << output_dispx(0,5) << " ";
      std::cout << output_dispz(0,3) << "\n";
    }

  // --- Write output file --- //
    f_dispx << tim(it);
    f_dispz << tim(it);
    for (size_t i = 0 ; i < fem.output_nnode ; i++) {
      f_dispx << " " << output_dispx(0,i);
      f_dispz << " " << output_dispz(0,i);
    }
    f_dispx << "\n";
    f_dispz << "\n";

    f_strxx << tim(it);
    f_strzz << tim(it);
    f_strxz << tim(it);
    for (size_t i = 0 ; i < fem.output_nelem ; i++) {
      f_strxx << " " << output_strxx(0,i);
      f_strzz << " " << output_strzz(0,i);
      f_strxz << " " << output_strxz(0,i);
    }
    f_strxx << "\n";
    f_strzz << "\n";
    f_strxz << "\n";
  }

  f_dispx.close();
  f_dispz.close();
  f_strxx.close();
  f_strzz.close();
  f_strxz.close();

  clock_t end = clock();
  std::cout << "elapsed_time: " << (double)(end - start) / CLOCKS_PER_SEC << "[sec]\n";
}
