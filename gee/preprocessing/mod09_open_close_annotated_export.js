// Center for Robust Decision Making on Climate and Energy Policy
// Clouds Project - Routine for data preprocessing
// WARNING: Google Earth Engine JS code (need to run using their client, or a Node.js env)

//TODO: Implement oceans crop

// Import section
//
var mod09ga = ee.ImageCollection("MODIS/006/MOD09GA"), // Raster collection
    closed_cell = /* color: #d63000 */ee.Geometry.MultiPoint(), // Polygon for closed field
    open_cell = /* color: #98ff00 */ee.Geometry.Polygon(
        [[[132.48494782929072, 29.772869636775006],
          [131.80478720948588, 20.305136254807795],
          [134.61760168375713, 22.433687586884023],
          [137.95810766244915, 22.677453130844274],
          [138.52965516805034, 19.93419149373989],
          [142.0898452858096, 21.04593381172043],
          [146.44135061314932, 24.33011086288961],
          [154.1337853393286, 27.023146597954202],
          [154.9253634164097, 29.958701433145098],
          [149.56251098345945, 29.768492645242056],
          [153.12333295519738, 32.06840859118038],
          [150.8814217102937, 31.919442514184453],
          [151.585065930547, 35.03535780611183],
          [148.5958530153448, 36.56856041052691],
          [144.37554922299864, 35.75234104568063],
          [141.1225301452298, 32.92151451909646],
          [133.51766064761853, 32.2032816545443]]]); // Polygon for open field

// Parameters ***********************************

// MODIS Collection
var modis_coll = 'MOD09GA';

// Date of the mosaic to be analyzed
var day = 12;
var month = 02;
var year = 2017;

var gDriveFolder = 'MOD09_open_closed_stratocumulus_oct022018_';
var cropLandMass = false; // Just keeps images from the oceans
// **********************************************

// generate start and end dates
var start_date = ee.Date.fromYMD(year,month,day);
var end_date = start_date.advance(1, 'day');  // Increments the start day in one day
var date_label = start_date.format('yyyy-MM-dd').getInfo();

var vizParams = {
  bands: ['sur_refl_b02', 'sur_refl_b01', 'sur_refl_b04'], // Define RGB composite in this order
  min: -100,
  max: 10000, // Use 8000 if needs to pop up visual response. Reflectance range between -100 and 16000.
  gamma: [1, 1, 1]
};

var bands_list = ['state_1km', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
var bands_list_export = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];

var vizImage = ee.Image(mod09ga.filterDate(start_date,end_date).select(bands_list).first());
// print(vizImage);
Map.addLayer(vizImage ,vizParams, 'ḾOD09GA');
print('Open cell stats:')
print('Area(km2): ',(ee.Number(open_cell.area()).divide(1000000)));
print('Perimeter(km): ',(ee.Number(open_cell.perimeter()).divide(1000)));
print('Approximate side(km): ',(ee.Number(open_cell.perimeter()).divide(4000)));

print('Closed cell stats:')
print('Area(km2): ', (ee.Number(closed_cell.area()).divide(1000000)));
print('Perimeter(km): ',(ee.Number(closed_cell.perimeter()).divide(1000)));
print('Approximate side(km): ',(ee.Number(closed_cell.perimeter()).divide(4000)));


// Mask function
function c_mask(image) {
  // Point to Bits 0 and 1
  var bit_0 = ee.Number(2).pow(0).int();
  var bit_1 = ee.Number(2).pow(1).int();

  // Get the cloud fraction.
  var cf = image.select('state_1km');

  // Cloud fraction
  var mask = cf.bitwiseAnd(bit_0).eq(0).and(  // cloudy
             cf.bitwiseAnd(bit_1).eq(1))  // cloudy

             .or(

             cf.bitwiseAnd(bit_0).eq(1)).and( // mixed
             cf.bitwiseAnd(bit_1).eq(0))      // mixed

             // If set to "not" removes clouds, otherwise keeps clouds
            ;
            // .not();

  // Return the masked image
  return image.updateMask(mask);
}

var filtered = ee.Image(mod09ga.filterDate(start_date,end_date).select(bands_list).map(c_mask).first());
Map.addLayer(filtered ,vizParams, 'Cloud segmented');

Export.image.toDrive({
  image: filtered.select(bands_list_export),
  description: date_label+'_'+modis_coll+'_closed-cell',
  folder: gDriveFolder+date_label,
  fileNamePrefix: date_label+'_'+modis_coll+'_closed-cell',
  region: closed_cell,
  scale: 500,
  crs: 'EPSG:4326'
});

Export.image.toDrive({
  image: filtered.select(bands_list_export),
  description: date_label+'_'+modis_coll+'_open-cell',
  folder: gDriveFolder+date_label,
  fileNamePrefix: date_label+'_'+modis_coll+'_open-cell',
  region: open_cell,
  scale: 500,
  crs: 'EPSG:4326'
});