#include <algorithm>
#include <chrono>
#include <string>
#include <mutex>
#include <unordered_set>

#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/math/Color.hh>
#include <gz/math/Temperature.hh>
#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Material.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Temperature.hh>
#include <sdf/Material.hh>

namespace warehouse_sim
{
class HazardHeatManager final : public gz::sim::System,
                                public gz::sim::ISystemConfigure,
                                public gz::sim::ISystemPreUpdate
{
  public: void Configure(const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &,
      gz::sim::EntityComponentManager &,
      gz::sim::EventManager &) override
  {
    this->node.Subscribe("/hazards/discovered",
        &HazardHeatManager::OnDiscovered, this);
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || _info.dt <= std::chrono::steady_clock::duration::zero())
      return;
    const double dt = std::chrono::duration<double>(_info.dt).count();
    _ecm.Each<gz::sim::components::Temperature>(
      [&](const gz::sim::Entity &_entity,
          const gz::sim::components::Temperature *_temperature)
      {
        const std::string modelName = this->HazardName(_entity, _ecm);
        if (modelName.empty())
          return true;
        {
          std::lock_guard<std::mutex> lock(this->mutex);
          if (this->discovered.count(modelName))
            return true;
        }

        const double oldK = _temperature->Data().Kelvin();
        const double newK = std::min(873.15, oldK + 2.0 * dt);
        if (newK > oldK + 1e-9)
        {
          gz::math::Temperature updated;
          updated.SetKelvin(newK);
          _ecm.SetComponentData<gz::sim::components::Temperature>(_entity, updated);
        }
        if (newK >= 523.15)
          this->MakeOrange(_entity, _ecm);
        return true;
      });
  }

  private: std::string HazardName(gz::sim::Entity _entity,
      const gz::sim::EntityComponentManager &_ecm) const
  {
    // Temperature belongs to the visual; walk up to its model to get the
    // exact instance name used by the detector's discovery event.
    gz::sim::Entity current = _entity;
    for (int depth = 0; depth < 4 && current != gz::sim::kNullEntity; ++depth)
    {
      const auto *name = _ecm.Component<gz::sim::components::Name>(current);
      if (name && name->Data().rfind("fire_hazard_", 0) == 0)
        return name->Data();
      const auto *parent = _ecm.Component<gz::sim::components::ParentEntity>(current);
      if (!parent)
        break;
      current = parent->Data();
    }
    return {};
  }

  private: void MakeOrange(gz::sim::Entity _entity,
      gz::sim::EntityComponentManager &_ecm)
  {
    if (!this->orange.insert(_entity).second)
      return;
    auto *material = _ecm.Component<gz::sim::components::Material>(_entity);
    if (!material)
      return;
    sdf::Material updated = material->Data();
    const gz::math::Color orange(1.0, 0.22, 0.015, 1.0);
    updated.SetAmbient(orange);
    updated.SetDiffuse(orange);
    updated.SetEmissive(gz::math::Color(0.9, 0.12, 0.0, 1.0));
    _ecm.SetComponentData<gz::sim::components::Material>(_entity, updated);
  }

  private: void OnDiscovered(const gz::msgs::StringMsg &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->discovered.insert(_msg.data());
  }

  private: gz::transport::Node node;
  private: std::mutex mutex;
  private: std::unordered_set<std::string> discovered;
  private: std::unordered_set<gz::sim::Entity> orange;
};
}

GZ_ADD_PLUGIN(warehouse_sim::HazardHeatManager,
  gz::sim::System, gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(warehouse_sim::HazardHeatManager,
  "warehouse_sim::HazardHeatManager")
