# AeonTerra Tectonics System Enhancement Plan

## Current State Analysis

Our current tectonics simulation implements:
- Basic plate creation and movement
- Simple boundary interactions (convergent, divergent, transform)
- Preliminary supercontinent cycle modeling
- Elementary climate-geology feedback

However, several aspects need improvement for more realistic terrain generation:

## 1. Plate Boundary Physics Enhancement

### Implementation Tasks

1. **Improve Subduction Mechanics**
   - Add proper slab pull calculations based on age and density of oceanic lithosphere
   - Implement variable subduction angles based on plate velocity and age
   - Model back-arc basin formation behind subduction zones
   ```cpp
   // New method for TectonicSimulation class
   float calculateSlabPull(int plateIdx, int boundaryVertexIdx) {
       float crustAge = planet.getCrustAge(boundaryVertexIdx);
       float density = planet.getCrustDensity(boundaryVertexIdx);
       float thickness = planet.getCrustThickness(boundaryVertexIdx);
       
       // Slab pull increases with age (denser, cooler slabs pull stronger)
       float basePull = 0.5f + 0.5f * std::min(1.0f, crustAge / 200.0f);
       // Denser crust creates more pull
       float densityFactor = (density - params.continentalDensity) / 
                           (params.oceanicDensity - params.continentalDensity);
       
       return basePull * densityFactor * thickness * params.tectonicActivityScale;
   }
   ```

2. **Isostatic Adjustment Refinement**
   - Implement multi-layer density model (crust, lithosphere, asthenosphere)
   - Add flexural isostasy for mountain ranges and subduction trenches
   - Model lithospheric bulging around mountain loads
   ```cpp
   // Enhanced isostatic calculation
   float calculateFlexuralAdjustment(int vertexIdx, float load) {
       const auto& neighbors = grid.getNeighbors(vertexIdx);
       float flexure = 0.0f;
       
       // Calculate flexural response based on elastic thickness of lithosphere
       float elasticThickness = planet.getLithosphereThickness(vertexIdx);
       float flexuralParameter = 50.0f * std::pow(elasticThickness / 25.0f, 3.0f);
       
       // Apply to neighboring regions with exponential decay
       for (int n : neighbors) {
           float distance = (grid.getVertex(vertexIdx) - grid.getVertex(n)).norm();
           float response = std::exp(-distance / flexuralParameter);
           flexure += response * load * 0.2f;
       }
       
       return flexure;
   }
   ```

3. **Transform Fault Mechanics**
   - Add strike-slip displacement along transform boundaries
   - Implement step-overs and pull-apart basin formation
   - Model transpression and transtension at fault bends
   ```cpp
   // Enhanced transform boundary processing
   void processTransformBoundary(int v1, int v2) {
       // Calculate relative motion along boundary
       Vec3f tangent = calculateBoundaryTangent(v1, v2);
       float slipRate = calculateRelativeMotion(v1, v2, tangent);
       
       // Apply offset based on slip rate
       float offset = slipRate * timeStep * 0.1f;
       
       // Generate pull-apart basins or pressure ridges at bends
       float bendAngle = calculateBoundaryBendAngle(v1, v2);
       if (std::abs(bendAngle) > 15.0f) {
           if (bendAngle > 0) {
               // Create transpressional ridge
               float elevation = planet.getElevation(v1);
               planet.setElevation(v1, elevation + 0.2f * offset);
           } else {
               // Create transtensional basin
               float elevation = planet.getElevation(v1);
               planet.setElevation(v1, elevation - 0.3f * offset);
           }
       }
   }
   ```

## 2. Mantle Plume Dynamics

### Implementation Tasks

1. **Hotspot Creation and Evolution**
   - Implement mantle plume generation with realistic spatial distribution
   - Model hotspot track formation as plates move over plumes
   - Create volcanic island chains along hotspot tracks
   ```cpp
   // Add to TectonicSimulation class
   class MantlePlume {
   public:
       MantlePlume(const Vec3f& position, float strength, float radius, float age = 0.0f)
           : position(position), strength(strength), radius(radius), age(age) {}
       
       Vec3f position;  // Position on sphere
       float strength;  // Plume strength
       float radius;    // Influence radius
       float age;       // Age in million years
       
       std::vector<int> affectedVertices;  // Vertices influenced by plume
   };
   
   std::vector<MantlePlume> mantlePlumes;
   
   void generateMantlePlumes(int count) {
       for (int i = 0; i < count; ++i) {
           Vec3f position = generateRandomSpherePoint();
           float strength = std::uniform_real_distribution<float>(0.5f, 2.0f)(rng);
           float radius = std::uniform_real_distribution<float>(100.0f, 300.0f)(rng);
           mantlePlumes.emplace_back(position, strength, radius);
       }
   }
   ```

2. **Large Igneous Province Formation**
   - Model large flood basalt events associated with plume heads
   - Implement rifting due to mantle plume impingement
   - Create plateau basalts and associated topography
   ```cpp
   void simulateLargeIgneousProvince(const MantlePlume& plume) {
       float baseElevation = 1.5f; // 1.5 km above sea level
       float baseThickness = 15.0f; // 15 km thickness
       float maxRadius = 500.0f; // 500 km radius
       
       for (int vertexIdx : plume.affectedVertices) {
           Vec3f pos = planet.getGrid().getVertex(vertexIdx);
           float distance = (pos - plume.position).norm();
           
           if (distance < maxRadius) {
               // Calculate elevation and thickness based on distance from center
               float factor = 1.0f - (distance / maxRadius);
               float elevation = baseElevation * factor * factor;
               float thickness = baseThickness + 10.0f * factor;
               
               // Set flood basalt properties
               planet.setElevation(vertexIdx, elevation);
               planet.setCrustThickness(vertexIdx, thickness);
               planet.setCrustType(vertexIdx, 2); // New crust type: flood basalt
           }
       }
   }
   ```

3. **Dynamic Topography**
   - Simulate lithospheric bulging due to plume upwelling
   - Model basin subsidence due to plume depletion
   - Create realistic volcanic edifices at hotspot locations
   ```cpp
   void updateDynamicTopography() {
       for (const auto& plume : mantlePlumes) {
           for (int vertexIdx : plume.affectedVertices) {
               Vec3f pos = planet.getGrid().getVertex(vertexIdx);
               float distance = (pos - plume.position).norm();
               
               // Calculate dynamic uplift based on distance and plume strength
               float uplift = plume.strength * std::exp(-distance / (plume.radius / 2.0f));
               
               // Apply dynamic topography
               float currentElevation = planet.getElevation(vertexIdx);
               planet.setElevation(vertexIdx, currentElevation + uplift * 0.05f);
           }
       }
   }
   ```

## 3. Supercontinent Cycle Integration

### Implementation Tasks

1. **Full Wilson Cycle Integration**
   - Integrate the existing SupercontinentCycle class into the main simulation
   - Add proper phase transitions with realistic timing
   - Implement continental configuration tracking
   ```cpp
   // Add to TectonicSimulation class constructor
   supercontinentCycle = std::make_unique<SupercontinentCycle>(*this);
   
   // Add to simulateStep
   void TectonicSimulation::simulateStep(double timeStep) {
       // Update supercontinent cycle
       supercontinentCycle->update(timeStep);
       
       // Rest of existing simulation code
       // ...
   }
   ```

2. **Supercontinent Breakup Mechanics**
   - Model rifting and breakup along zones of weakness
   - Implement realistic sea floor spreading during dispersal phases
   - Create passive margin formation sequence
   ```cpp
   void simulateSupercontinentBreakup() {
       // Find continental plates
       std::vector<int> continentalPlates;
       for (size_t i = 0; i < plates.size(); ++i) {
           if (!plates[i].isOceanic()) {
               continentalPlates.push_back(i);
           }
       }
       
       // Calculate continental centroid
       Vec3f centroid = calculateContinentalCentroid(continentalPlates);
       
       // Create rift zones along supercontinent
       for (int plateIdx : continentalPlates) {
           const auto& vertices = plates[plateIdx].getVertices();
           
           // Find vertices far from center of supercontinent
           for (int vertexIdx : vertices) {
               Vec3f pos = planet.getGrid().getVertex(vertexIdx);
               float distanceToCentroid = (pos - centroid).norm();
               
               // Potential rift locations are far from centroid
               if (distanceToCentroid > plateau.getRadius() * 0.7f) {
                   float chance = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                   if (chance < 0.02f) {
                       // Create a rift
                       createRiftZone(vertexIdx, plateIdx);
                   }
               }
           }
       }
   }
   ```

3. **Assembly Phase Collision Improvements**
   - Enhance mountain building during continent-continent collisions
   - Implement realistic suture zone formation
   - Add crust thickening and deformation during collision
   ```cpp
   void enhanceCollisionBoundary(int v1, int v2) {
       int p1 = planet.getPlateId(v1);
       int p2 = planet.getPlateId(v2);
       
       // Only enhance continental-continental collisions
       if (!plates[p1].isOceanic() && !plates[p2].isOceanic()) {
           // Calculate collision intensity based on relative velocities
           Vec3f vel1 = plates[p1].getMovement();
           Vec3f vel2 = plates[p2].getMovement();
           Vec3f relVel = vel1 - vel2;
           float intensity = relVel.norm() * 2.0f;
           
           // Thicken crust at collision zone
           float thickness1 = planet.getCrustThickness(v1);
           float thickness2 = planet.getCrustThickness(v2);
           float newThickness = std::min(70.0f, (thickness1 + thickness2) * 1.2f);
           
           // Thicken both sides
           planet.setCrustThickness(v1, newThickness);
           planet.setCrustThickness(v2, newThickness);
           
           // Create mountain range
           float elevation1 = planet.getElevation(v1);
           float elevation2 = planet.getElevation(v2);
           float newElevation = std::max(elevation1, elevation2) + intensity * 0.5f;
           
           // Cap mountain height based on isostasy
           newElevation = std::min(9.0f, newElevation);
           
           planet.setElevation(v1, newElevation);
           planet.setElevation(v2, newElevation * 0.9f);
       }
   }
   ```

## 4. Climate-Geology Feedback

### Implementation Tasks

1. **Enhanced Weathering Model**
   - Implement chemical weathering based on temperature and precipitation
   - Model physical weathering based on temperature variations and elevation
   - Create realistic erosion patterns in different climate zones
   ```cpp
   float calculateWeatheringRate(int vertexIdx) {
       // Get climate data
       float temperature = climate.temperature[vertexIdx];
       float precipitation = climate.precipitation[vertexIdx];
       float elevation = planet.getElevation(vertexIdx);
       
       // Base chemical weathering (stronger in warm, wet areas)
       float chemicalWeathering = 0.0f;
       if (temperature > 0.0f) {
           // Arrhenius-like equation for temperature dependency
           chemicalWeathering = 0.1f * std::exp(temperature / 20.0f) * 
                               (precipitation / 2000.0f);
       }
       
       // Physical weathering (stronger at high elevations and freeze-thaw zones)
       float physicalWeathering = 0.0f;
       if (elevation > 0.0f) {
           physicalWeathering = 0.05f * elevation;
           
           // Enhanced freeze-thaw weathering near 0°C
           if (temperature > -5.0f && temperature < 5.0f) {
               physicalWeathering *= 2.0f;
           }
       }
       
       return chemicalWeathering + physicalWeathering;
   }
   ```

2. **Orographic Precipitation**
   - Model rainfall patterns influenced by topography
   - Implement rain shadow effects on leeward side of mountains
   - Create realistic moisture distribution based on prevailing winds
   ```cpp
   void calculateOrographicPrecipitation() {
       // Simplistic wind direction based on latitude (trade winds, westerlies, etc.)
       std::vector<Vec3f> windDirections(planet.getVertexCount());
       
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           Vec3f pos = planet.getGrid().getVertex(i);
           float lat = planet.getGrid().cartesianToLatLon(pos).first;
           
           // Simple wind model (easterly trade winds near equator, westerlies in mid-latitudes)
           if (std::abs(lat) < 30.0f) {
               // Trade winds (easterly)
               windDirections[i] = Vec3f(-1.0f, 0.0f, 0.0f);
           } else if (std::abs(lat) < 60.0f) {
               // Westerlies
               windDirections[i] = Vec3f(1.0f, 0.0f, 0.0f);
           } else {
               // Polar easterlies
               windDirections[i] = Vec3f(-1.0f, 0.0f, 0.0f);
           }
           
           // Project onto tangent plane
           Vec3f normal = pos.normalized();
           windDirections[i] = windDirections[i] - normal * windDirections[i].dot(normal);
           windDirections[i].normalize();
       }
       
       // Calculate precipitation changes due to orography
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           int vertexIdx = static_cast<int>(i);
           const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
           
           float upwindGradient = 0.0f;
           
           for (int neighborIdx : neighbors) {
               Vec3f toNeighbor = planet.getGrid().getVertex(neighborIdx) - 
                                 planet.getGrid().getVertex(vertexIdx);
               toNeighbor.normalize();
               
               // Check if this neighbor is upwind
               float dotProduct = toNeighbor.dot(windDirections[i]);
               if (dotProduct > 0.3f) {
                   // This is relatively upwind
                   float elevationDiff = planet.getElevation(neighborIdx) - 
                                       planet.getElevation(vertexIdx);
                   upwindGradient += elevationDiff;
               }
           }
           
           // Modify precipitation based on upwind gradient
           if (upwindGradient < 0.0f) {
               // Air rising up slope - increased precipitation
               climate.precipitation[i] *= (1.0f - upwindGradient);
           } else if (upwindGradient > 0.0f) {
               // Air descending - rain shadow effect
               climate.precipitation[i] *= 1.0f / (1.0f + upwindGradient);
           }
       }
   }
   ```

3. **Sediment Transport and Deposition**
   - Implement realistic fluvial sediment transport
   - Model delta formation at river mouths
   - Create coastal sediment distribution
   ```cpp
   void simulateSedimentTransport() {
       // Calculate sediment generated through weathering
       std::vector<float> sedimentProduction(planet.getVertexCount(), 0.0f);
       std::vector<float> sedimentAccumulation(planet.getVertexCount(), 0.0f);
       
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           int vertexIdx = static_cast<int>(i);
           float weatheringRate = calculateWeatheringRate(vertexIdx);
           sedimentProduction[i] = weatheringRate * params.erosionRate * static_cast<float>(timeStep);
       }
       
       // Transport sediment downstream based on flow directions
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           int vertexIdx = static_cast<int>(i);
           int downstream = hydrology.flowDirection[i];
           
           if (downstream >= 0) {
               // Transport 80% of sediment downstream
               sedimentAccumulation[downstream] += sedimentProduction[i] * 0.8f;
               sedimentAccumulation[i] += sedimentProduction[i] * 0.2f;
           } else {
               // No downstream - all sediment deposits locally (basin)
               sedimentAccumulation[i] += sedimentProduction[i];
           }
       }
       
       // Apply sediment accumulation to elevation and crust thickness
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           int vertexIdx = static_cast<int>(i);
           float elevation = planet.getElevation(vertexIdx);
           float thickness = planet.getCrustThickness(vertexIdx);
           
           planet.setElevation(vertexIdx, elevation + sedimentAccumulation[i]);
           planet.setCrustThickness(vertexIdx, thickness + sedimentAccumulation[i]);
       }
       
       // Special case: river mouths form deltas
       for (size_t i = 0; i < planet.getVertexCount(); ++i) {
           int vertexIdx = static_cast<int>(i);
           if (hydrology.isRiver[i] && planet.getElevation(vertexIdx) < 0.1f) {
               // This is near sea level and has a river - potential delta
               float riverFlow = hydrology.riverFlow[i];
               if (riverFlow > 10.0f) {
                   // Significant river - form delta
                   float deltaSize = std::log10(riverFlow) * 0.1f;
                   
                   // Apply delta to neighboring low elevation cells
                   const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
                   for (int neighborIdx : neighbors) {
                       if (planet.getElevation(neighborIdx) < 0.0f) {
                           float currentElev = planet.getElevation(neighborIdx);
                           planet.setElevation(neighborIdx, 
                                             std::min(0.05f, currentElev + deltaSize));
                       }
                   }
               }
           }
       }
   }
   ```

## 5. Implementation Strategy

### Phase 1: Core Tectonics Enhancements (3-4 weeks)
1. Extend the Planet class with additional properties
   - Add lithospheric thickness property
   - Add crust age property
   - Add flex parameter property
2. Implement improved subduction mechanics
3. Enhance isostatic adjustment calculations
4. Improve transform boundary processing

### Phase 2: Mantle Plume System (2-3 weeks)
1. Create MantlePlume class and integration
2. Implement hotspot track generation
3. Add large igneous province formation
4. Implement dynamic topography

### Phase 3: Supercontinent Cycle Integration (2-3 weeks)
1. Fully integrate the SupercontinentCycle class
2. Implement supercontinent breakup mechanics
3. Enhance collision processes
4. Add tracking of continental configurations

### Phase 4: Climate-Geology Feedback (3-4 weeks)
1. Implement enhanced weathering model
2. Add orographic precipitation calculation
3. Create sediment transport and deposition system
4. Implement delta and coastal feature formation

### Phase 5: Testing and Optimization (2 weeks)
1. Implement unit tests for each new component
2. Create integration tests for the full system
3. Optimize performance bottlenecks
4. Document the enhanced tectonics system

## Conclusion

The proposed enhancements will significantly improve the realism of our tectonic simulations and heightmap generation. The implementation follows a phased approach, prioritizing the core tectonics enhancements first, then building additional systems on top. This strategy allows for incremental testing and validation of each component.

By focusing on the physics-based approach to tectonics, we'll create a system that accurately models Earth-like processes while being flexible enough to simulate alternative planetary conditions.