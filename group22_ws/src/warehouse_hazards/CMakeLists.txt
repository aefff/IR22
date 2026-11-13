cmake_minimum_required(VERSION 3.8)
project(warehouse_hazards)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(sdformat14 REQUIRED)

add_library(hazard_heat_manager SHARED src/hazard_heat_manager.cc)
target_compile_features(hazard_heat_manager PRIVATE cxx_std_17)
target_link_libraries(hazard_heat_manager PRIVATE
  gz-sim8::gz-sim8 gz-plugin2::register gz-transport13::gz-transport13
  gz-math7::gz-math7 gz-msgs10::gz-msgs10 sdformat14::sdformat14)

install(TARGETS hazard_heat_manager LIBRARY DESTINATION lib)
ament_package()
