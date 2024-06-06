from numpy import cos, deg2rad
from scipy.optimize import newton, toms748

class Location:
    def __init__(self, distance_from_sun_au) -> None:
        self.distance_from_sun_au = distance_from_sun_au
        self.q_S = 1380.73 / (distance_from_sun_au**2)


class Surface:
    stefan_boltzmann_constant = 5.670374419e-8

    def __init__(
        self,
        emission_area,
        sun_facing_area,
        emissivity          = 0.5,
        absorptance         = 0.5,
        angle_to_sun_deg    = 0,
        temperature_offset  = 0,
    ) -> None:
        """
        emission_area is the radiative area. This is often both sides of a plate.\n
        sun_facing_area is the area that solar heating will act on. This is only ever one side of a plate.\n
        emissivity is what is says it is.\n
        absorptance is the solar absorptance.\n
        angle_to_sun_deg is the angle of the surface to the sun. An angle of 0 degrees means that the normal to the surface is pointing directly at the sun. An angle of 90 degrees means that the plate is edge on and not illuminated.\n
        temperature_offset is an offset in the isothermal temperature of the panel relative to the perceived temperature of the system. This is useful for representing active cooling of solar panels.
        """
        self.emissivity         = emissivity
        self.absorptance        = absorptance
        self.emission_area      = emission_area
        self.sun_facing_area    = sun_facing_area
        self.sun_facing_area   *= cos(deg2rad(angle_to_sun_deg)) if angle_to_sun_deg else 1
        self.temperature_offset = temperature_offset
        self.T                  = None

    def set_T(self, T):
        self.T = T + self.temperature_offset

    ##--Q_E
    def radiated_heat(self, T):
        T_perceived = T + self.temperature_offset
        return self.emissivity * self.stefan_boltzmann_constant * self.emission_area * (T_perceived**4)

    ##--Q_S
    def direct_solar_thermal_heat_received(self, loc : Location):
        return loc.q_S * self.absorptance * self.sun_facing_area

    def __repr__(self):
        return f"Temperature = {self.T:.1f} K"


def clean_surfaces_input(surfaces : Surface | list[Surface] | dict[str : Surface]):
    if isinstance(surfaces, Surface):
        surfaces = [surfaces]
    elif isinstance(surfaces, dict):
        surfaces = surfaces.values()
    elif isinstance(surfaces, list):
        pass
    else:
        raise ValueError("Invalid format for 'surfaces' or 'radiators': must be a Surface, list of Surfaces, or a dict of Surfaces. Quitting.")
    return surfaces


class HeatSource:
    def __init__(self, q_w = 0) -> None:
        self._q_w = q_w
        self.T_C = None

    def q_w(self, *args):
        return self._q_w

    def __repr__(self):
        return f"Cold-Side Temperature = {self.T_C:.1f} K\nWaste Heat = {self.q_w():.1f} W"

class PowerSource(HeatSource):
    def __init__(self, w = None, q_w = None) -> None:
        self._w = w
        super().__init__(q_w)

    def w(self, *args):
        return self._w

    def __repr__(self):
        return super().__repr__() + f"\nUseful Power = {self.w():.1f} W"


class Turbine(PowerSource):
    eta_p = 0.70

    def __init__(self, qt, T_H) -> None:
        self.qt     = qt
        self.T_H    = T_H

        super().__init__(0,0)

    def eta_C(self, T_C):
        return 1 - T_C / self.T_H if T_C < self.T_H else 0.0

    def eta_T(self, T_C):
        return self.eta_p * self.eta_C(T_C)

    def w(self, T_C = None):
        if T_C == None:
            T_C = self.T_C
        return self.qt * self.eta_T(T_C)

    def q_w(self, T_C = None):
        if T_C == None:
            T_C = self.T_C
        return self.qt - self.w(T_C)

    def __repr__(self):
        return super().__repr__() + f"\nHot-Side Temperature = {self.T_H:.1f} K"


class Reactor(Turbine):
    reactors = {
        "fission_frontiers" : {
            "tarasque"  : {"qt" : 1e9,   "T_H" : 1300},
            "guivre"    : {"qt" : 2e9,   "T_H" : 1100},
            "peluda"    : {"qt" : 1e9,   "T_H" : 600},
            "lindworm"  : {"qt" : 500e6, "T_H" : 950},
            "wyvern"    : {"qt" : 250e6, "T_H" : 600},
        },
        "fusion_founders" : {
            "standard_reactor" : {"qt" : 300e6, "T_H" : 1300},
        }
    }

    def __init__(
        self,
        qt = None,
        T_H = None,
        brand = None,
        model = None,
    ) -> None:
        """
        qt is the thermal power in Watts if you are entering a custom reactor specification.\n
        T_H is the hot reservoir temperature in Kelvin if you are entering a custom reactor specification.\n
        brand is the name of the reactor manufacturer for a standard installation. Either "fission_frontiers" or "fusion_founders".\n
        model is the name of the reactor model for a standard installation. Fission Frontiers offer: "tarasque", "guivre", "peluda", "lindworm", "wyvern", while Fusion Founders offer: "standard_reactor" only.
        """
        custom  = qt and T_H
        library = brand and model
        if not custom and not library:
            raise ValueError("Reactor is not adequately defined: both of 'qt' and 'T_H' or both of 'brand' and 'model' must be complete! Quitting.")
        elif custom and library:
            raise ValueError("Reactor is over defined: both of 'qt' and 'T_H' or both of 'brand' and 'model' must be complete, not all of them! Quitting.")
        elif custom:
            super().__init__(qt, T_H)
        elif library:
            try:
                reactor_data = self.reactors[brand.lower()][model.lower()]
                super().__init__(reactor_data["qt"], reactor_data["T_H"])
            except:
                lines = [
                    f"Reactor '{model}' by '{brand}' cannot be found! The valid brands and their models are:"
                ]
                for brand in self.reactors.keys():
                    lines.append(f"\t{brand}:")
                    for model in self.reactors[brand].keys():
                        lines.append(f"\t\t{model}")
                raise ValueError("\n".join(lines))
        else:
            raise ValueError()


class SolarAbsorber(Turbine):
    def __init__(
        self,
        loc,
        qt : float | int | list[float] | list[int] | dict[str : float] | dict[str : int] = [],
        surfaces : Surface | list[Surface] | dict[str : Surface] = [],
    ) -> None:
        sum_qt = sum( qt if isinstance(qt, list) else [qt] )

        if isinstance(qt, float) or isinstance(qt, int):
            sum_qt = qt
        elif isinstance(qt, dict):
            sum_qt = sum( [ v for v in qt.values()] )
        elif isinstance(qt, list):
            sum_qt = sum( qt )
        else:
            raise ValueError("Invalid format for 'surfaces': must be a Surface, list of Surfaces, or a dict of Surfaces. Quitting.")

        self.surfaces = clean_surfaces_input(surfaces)

        sum_qs = sum( [surf.direct_solar_thermal_heat_received(loc) for surf in self.surfaces] )

        if sum_qt > sum_qs:
            raise ValueError(f"The sum of qt (~{sum_qt:.0f} W), the power pulled from the solar absorber, is greater than the energy that it is receiving (~{sum_qs:.0f}). Quitting.")

        def absorber_balance(T_H):
            nonlocal loc, qt, self
            Q_W = sum_qt
            Q_S = sum_qs
            Q_E = sum( [surf.radiated_heat(T_H) for surf in self.surfaces] )
            return Q_E - Q_W - Q_S

        T_H = newton(absorber_balance, 273.15, tol = 1e-3, maxiter = 2000)
        self.q_incident = sum( [surf.direct_solar_thermal_heat_received(loc) for surf in self.surfaces] )
        self.q_emitted  = sum( [surf.radiated_heat(T_H) for surf in self.surfaces] )
        for surf in self.surfaces:
            surf.T = T_H

        super().__init__(sum_qt, T_H)


class Photovoltaic(PowerSource):
    non_optimal_illumination_factor = 0.8
    c_a_at_earth                    = -3.4e-2
    c_t                             = -0.2e-2
    reflection_fraction             = 5e-3
    temperature_limits              = {"min" : -100 + 273.15, "max" : 150 + 273.15}

    def __init__(
        self,
        loc,
        eta_lab,
        T_ref,
        age_years,
        surfaces : Surface | list[Surface] | dict[str : Surface] = [],
    ) -> None:

        self.loc        = loc
        self.eta_lab    = eta_lab
        self.T_ref      = T_ref
        self.age_years  = age_years
        self.surfaces   = clean_surfaces_input(surfaces)

        for surf in self.surfaces:
            if surf.absorptance != 0:
                print("Setting absorptance of surface assigned to photovoltaic power source to zero to prevent double counting of heat load. Remember to include this surface as a radiator in the ThermalLoop!")
                surf.absorptance = 0

        self.total_collection_area  = sum( [surf.sun_facing_area for surf in self.surfaces] )
        self.total_radiative_area   = sum( [surf.emission_area for surf in self.surfaces] )
        super().__init__()


    def eta_N(self):
        return self.non_optimal_illumination_factor * self.eta_lab

    def c_a(self):
        return self.c_a_at_earth / (self.loc.distance_from_sun_au**2)

    def eta_overall(self, T_C):
        if T_C <= self.temperature_limits["min"]:
            return 0.0
        elif T_C >= self.temperature_limits["max"]:
            return 0.0
        if not "eta_non_thermal" in self.__dict__:
            self.eta_non_thermal = self.eta_N() + self.age_years * self.c_a()
            self.temperature_limits["max"] = (- self.eta_non_thermal / self.c_t) + self.T_ref
            print(f"Panel max temperature = {self.temperature_limits['max']:.1f} K ({self.temperature_limits['max'] - 273.15:.1f} degC)")
        return self.eta_non_thermal + self.c_t * (T_C - self.T_ref)

    def w(self, T_C = None):
        if T_C == None:
            T_C = self.T_C
        return self.eta_overall(T_C) * self.loc.q_S * self.total_collection_area

    def q_w(self, T_C = None):
        if T_C == None:
            T_C = self.T_C
        return (1 - self.reflection_fraction) * (1 - self.eta_overall(T_C)) * self.loc.q_S * self.total_collection_area

    def __repr__(self):
        return "\n\n".join([str(surf) for surf in self.surfaces]) + f"\nUseful Power = {self.w():.1f} W\nWaste Heat = {self.q_w():.1f} W"


class ThermalLoop:
    def __init__(
        self, 
        loc : Location, 
        heat_sources : HeatSource | list[HeatSource] |dict[str : HeatSource] = {}, 
        radiators : Surface | list[Surface] | dict[str : Surface] = [],
    ) -> None:
        self.loc    = loc
        self._T_C   = None

        if isinstance(heat_sources, HeatSource):
            self.heat_sources = [heat_sources]
        elif isinstance(heat_sources, dict):
            self.heat_sources = heat_sources.values()
        elif isinstance(heat_sources, list):
            self.heat_sources = heat_sources
        else:
            raise ValueError("Invalid format for 'heat_sources': must be a HeatSource, list of HeatSources, or a dict of HeatSources. Quitting.")

        self.radiators = clean_surfaces_input(radiators)

        hsT_H = [hs.T_H for hs in self.heat_sources if "T_H" in hs.__dict__]
        hsT_H.append(10e3)
        self.min_T_H = min(hsT_H)

    @property
    def T_C(self):
        if not self._T_C:
            self._T_C = self.equilibrate()
        return self._T_C

    def __radiator_balance(self, T_C):
        Q_W = sum( [hs.q_w(T_C) for hs in self.heat_sources] )
        Q_S = sum( [surf.direct_solar_thermal_heat_received(self.loc) for surf in self.radiators] )
        Q_E = sum( [surf.radiated_heat(T_C) for surf in self.radiators] )
        return Q_E - Q_W - Q_S

    def equilibrate(self, initial_value = 273.15, tol = 1e-3):
        try:
            self._T_C = toms748(self.__radiator_balance, 0, self.min_T_H, rtol = tol)
        except ValueError:
            raise ValueError("A radiator temperature cannot be found that is less than the minimum T_H in the system. You need a larger, more emissive, or less absorptive radiator. Quitting.")

        for surf in self.radiators:
            surf.set_T(self.T_C)
        for hs in self.heat_sources:
            hs.T_C = self.T_C
        return self.T_C

    def __repr__(self):
        return "heat sources:\n" + "\n\n".join([str(heat_source) for heat_source in self.heat_sources]) + "\n\nradiators:\n" + "\n".join([str(rad) for rad in self.radiators])


if __name__ == "__main__":

    ##--Define the location in space
    loc = Location(distance_from_sun_au = 0.25)

    ##--Thermal Loop 1
    ##----Define power sources:
    ##------Define a nuclear reactor:
    reactor_1   = Reactor(brand = "fission_frontiers", model = "wyvern")

    ##------Define a Solar-Thermal Power system
    absorber_surface_1  = Surface(emission_area = 1000, sun_facing_area = 1000, emissivity = 0.12, absorptance = 0.96, angle_to_sun_deg = 25)
    absorber_surface_2  = Surface(emission_area = 200, sun_facing_area = 200, emissivity = 0.12, absorptance = 0.96, angle_to_sun_deg = 20)
    solar_thermal_1     = SolarAbsorber(loc, 1e6, [absorber_surface_1, absorber_surface_2])

    ##----Define the shared radiators
    radiator_1  = Surface(2 * 1000**2, 1000**2, 0.9, 0.09, 90)
    radiator_2  = Surface(2 * 800**2, 800**2, 0.9, 0.09, 70, temperature_offset = -20)
    radiator_3  = Surface(2 * 800**2, 800**2, 0.9, 0.09, 70, temperature_offset = -20)

    ##----Assemble a thermal-loop
    thermal_loop_1 = ThermalLoop(loc, heat_sources = [reactor_1, solar_thermal_1], radiators = [radiator_1, radiator_2, radiator_3])
    thermal_loop_1.equilibrate()

    ##--Printing out the standard (rather inconvenient) format for a thermal loop object
    print("thermal_loop_1:")
    print(thermal_loop_1)

    ##--Thermal Loop 2
    ##----Define power sources for Thermal Loop 2:
    pv_surface_1    = Surface(2 * 100, 100, 0.92, 0.12)
    photovoltaic_1  = Photovoltaic(loc, 0.33, 25 + 273.25, 0, pv_surface_1)

    ##--The photovoltaic panel surface will act as its own radiator to an extent, but it is still necessary to tell ThermalLoop that it will act thus by passing it in.
    ##--You can also add additional radiators to your solar panels, for example via an active cooling circuit. We include a negative temperature offset to represent that the panels will be hotter than an external radiator since the heat will have to be moved from the panel to the radiators
    pv_rad = Surface(2000, 1000, 0.78, 0.78, 90, temperature_offset = -10)

    thermal_loop_2 = ThermalLoop(loc, heat_sources = photovoltaic_1, radiators = [pv_surface_1, pv_rad])
    thermal_loop_2.equilibrate()

    ##--You can also print out specific objects to make the data easier to understand
    print("photovoltaic_1:")
    print(photovoltaic_1)
    print()

    print("pv_rad:")
    print(pv_rad)
    print()

    ##--Operational Loop 1
    atmospheric_heat_load_1 = HeatSource(100000)
    atmospheric_heat_load_2 = HeatSource(200000)

    operational_loop_rad_1 = Surface(2000, 1000, 0.92, 0.12, angle_to_sun_deg = 90, temperature_offset = -20) ##--The sun facing area is half, but will actually fall to zero because it is at 90 degrees to the sun

    operational_loop_1 = ThermalLoop(loc, [atmospheric_heat_load_1, atmospheric_heat_load_2], [operational_loop_rad_1])
    operational_loop_1.equilibrate()

    print("operational_loop_1:")
    print(operational_loop_1)
    ##--We can see that this radiator is too big for this heat load, as the Cold-Side Temperature perceived by the heat sources is 251.6 K (-23.6 degC) - a little chilly for your residents!