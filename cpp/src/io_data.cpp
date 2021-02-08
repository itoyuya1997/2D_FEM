#include "all.h"
#include "node.h"
#include "element.h"
#include "material.h"
#include "fem.h"

// ------------------------------------------------------------------- //
Fem input_mesh (const std::string mesh_file) {
  int nnode, nelem, nmaterial, dof;
  std::string line;

  // Open file //
  std::ifstream f(mesh_file);

  // Read header //
  std::getline(f, line);
  std::istringstream iss(line);
  iss >> nnode >> nelem >> nmaterial >> dof;

  // Read nodes //
  std::vector<Node> nodes;
  for (int inode = 0 ; inode < nnode ; inode++) {
    int id;
    std::vector<double> xyz(2);
    std::vector<int> freedom(dof);

    std::getline(f, line);
    // std::cout << line + "\n";

    std::istringstream iss(line);
    iss >> id;
    for (int i = 0 ; i < 2 ; i++) {
      iss >> xyz.at(i);
    }
    for (int i = 0 ; i < dof ; i++) {
      iss >> freedom.at(i);
    }

    Node node(id,xyz,freedom);
    nodes.push_back(node);
  }

  // Read elements //
  std::vector<Element> elements;
  for (int ielem = 0 ; ielem < nelem ; ielem++) {
    int id;
    std::string style;
    int material_id;
    std::vector<int> inode;

    std::getline(f, line);
    // std::cout << line + "\n";

    std::istringstream iss(line);
    iss >> id >> style >> material_id ;
    while(!iss.eof()) {
      int in;
      iss >> in;
      inode.push_back(in);
    }

    Element element(id,style,material_id,inode);
    elements.push_back(element);
  }

  // Read materials //
  std::vector<Material> materials;
  for (int imaterial = 0 ; imaterial < nmaterial ; imaterial++) {
    int id;
    std::string style;
    std::vector<double> param;

    std::getline(f, line);
    // std::cout << line + "\n";

    std::istringstream iss(line);
    iss >> id >> style ;
    while(!iss.eof()) {
      double ip;
      iss >> ip;
      param.push_back(ip);
    }

    Material material(id,style,param);
    materials.push_back(material);
  }

  Fem fem(dof,nodes,elements,materials);
  return fem;
}