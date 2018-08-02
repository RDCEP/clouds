// Center for Robust Decision Making on Climate and Energy Policy
// Clouds Project - Routine for data export
// Attention: This routine runs on Google Earth Engine JS API

// Parameters ***********************************

// MODIS Collection
var modis_coll = 'MOD09GA'; // GEE Modis Collection

// Date of the mosaic to be analyzed
var day = 1;
var month = 01;
var year = 2017;

var int_cf_threshold = 50 // Lower bound percentage threshold of cloud fraction (integer)
var cf_threshold = int_cf_threshold/100;

var grid_size = 10; // grid cell size in degrees (integer)
var gDriveFolder = '2017_january';
// **********************************************

//////////////////////////////////////////////////////////////////////////////////////////
// Generate grid stats
//////////////////////////////////////////////////////////////////////////////////////////

/***
 * Generates a regular grid using given bounds, specified as geometry.
 */
var generateGrid = function(xmin, ymin, xmax, ymax, dx, dy, marginx, marginy, opt_proj) {
    var proj = opt_proj || 'EPSG:4326'

    var xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(ee.Number(dx).multiply(0.9)), dx)
    var yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(ee.Number(dy).multiply(0.9)), dy)

    var cells = xx.map(function(x) {
        return yy.map(function(y) {
            var x1 = ee.Number(x).subtract(marginx)
            var x2 = ee.Number(x).add(ee.Number(dx)).add(marginx)
            var y1 = ee.Number(y).subtract(marginy)
            var y2 = ee.Number(y).add(ee.Number(dy)).add(marginy)

            var coords = ee.List([x1, y1, x2, y2]);
            var rect = ee.Algorithms.GeometryConstructors.Rectangle(coords, proj, false);

            return ee.Feature(rect)
        })
    }).flatten();

    return ee.FeatureCollection(cells);
}



var grid = generateGrid(-180, -60, 180, 60, grid_size, grid_size, 0, 0)

// generate start and end dates
var start_date = ee.Date.fromYMD(year,month,day);
var end_date = start_date.advance(1, 'day');  // Increments the start day in one day


var modis_stats = ee.ImageCollection('MODIS/006/'+modis_coll)
    .select(['sur_refl_b02', 'sur_refl_b01', 'sur_refl_b04', 'state_1km'])
    .filterDate(start_date,end_date);

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

var filtered = modis_stats.map(c_mask);
var clouds = ee.Image(filtered.first());

function joint_count(image){
    return image.rename('cloud_count').addBands(image.unmask().rename('total_count')).reduceRegions({
            collection: grid,
            reducer: ee.Reducer.count(),
            scale: 1000 // Counts made in 1km resolution, due to the resolution of the quality band
        }
    )
}

function cloud_fraction(featcollection){
    var cfrac = ee.Number(featcollection.get('cloud_count'))
    cfrac = cfrac.divide(ee.Number(featcollection.get('total_count')))
    return featcollection.set('cloud_fraction', cfrac)
}


var combined = ee.ImageCollection(clouds.select('state_1km'))
    .map(joint_count).flatten() // Counts number of pixels per cell and the cloudy
    .map(cloud_fraction) // Calculates cf ratio


//////////// Grid passed to Export routine
var table = combined.filter(ee.Filter.greaterThanOrEquals('cloud_fraction', cf_threshold))


//////////////////////////////////////////////////////////////////////////////////////////
// End grid stats
//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////
// Prepare image Export using thresholding
//////////////////////////////////////////////////////////////////////////////////////////
var modis = ee.Image(ee.ImageCollection('MODIS/006/'+modis_coll)
    .filterDate(start_date,end_date)
    .select(['state_1km','sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'], ['state_1km','b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
    .first()
)
    .clipToCollection(table);

var proj = modis.projection();

var vizParams = {
    bands: ['b2', 'b1', 'b4'],
    min: -100,
    max: 8000, // Use 8000 if needs to pop up visual response. Reflectance range between -100 and 16000.
    gamma: [1, 1, 1]
};




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

var filtered = ee.ImageCollection(modis).map(c_mask);


var constant = ee.Image.constant(0);
function missing_val_const(image){
    // For each image, imput random values

    var result = constant.blend(image);

    return result;
}

var zero_inputated = filtered.map(missing_val_const);

var random = ee.Image.random().multiply(16100).subtract(100).toInt16();
function missing_val_rand(image){
    // For each image, imput random values

    var result = random.blend(image);

    return result;
}

var random_inputated = filtered.map(missing_val_rand);

// /////// Print cloud fraction stats (uncomment if inspecting)

// print('Generated Polygons with stats')
// print(combined)

// print('Polygons with cloud fraction higher than '+cf_threshold)
// print(table)

// /////// Plot Maps (uncomment if inspecting)

// Map.addLayer(modis, vizParams, 'Images filtered by threshold')
// Map.addLayer(filtered, vizParams, 'Background removal')
// Map.addLayer(zero_inputated, vizParams, 'Missing values zero imputated')
// Map.addLayer(random_inputated, vizParams, 'Missing values random imputated')

//////// Layer Export
var date_label = start_date.format('yyyy-MM-dd').getInfo();
var global_region = ee.Geometry.Polygon([-180, 60, 0, 60, 180, 60, 180, -60, 10, -60, -180, -60], null, false);

Export.image.toDrive(

    {   // Export raw 7 bands, of regions higher than cf threshold
        image: ee.Image(modis.select(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])),
        description: date_label+'_'+modis_coll+'_raw_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        folder: gDriveFolder,
        fileNamePrefix: date_label+'_'+modis_coll+'_raw_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        scale: 500,
        maxPixels: 1e11,
        crs: 'EPSG:4326',
        region: global_region
    }
);

Export.image.toDrive(

    {   // Export 7 bands image, with background removal, of regions higher than cf threshold
        image: ee.Image(filtered.select(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']).first()),
        description: date_label+'_'+modis_coll+'_background_removal_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        folder: gDriveFolder,
        fileNamePrefix: date_label+'_'+modis_coll+'_background_removal_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        scale: 500,
        maxPixels: 1e11,
        crs: 'EPSG:4326',
        region: global_region
    }
);

Export.image.toDrive(

    {   // Export 7 bands image, with background removal, and values inputated with an integer, of regions higher than cf threshold
        image: ee.Image(zero_inputated.select(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']).first()),
        description: date_label+'_'+modis_coll+'_background_removal_zero_inputated_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        folder: gDriveFolder,
        fileNamePrefix: date_label+'_'+modis_coll+'_background_removal_zero_inputated_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        scale: 500,
        maxPixels: 1e11,
        crs: 'EPSG:4326',
        region: global_region
    }
);

Export.image.toDrive(

    {   // Export 7 bands image, with background removal, and values inputated with random normally numbers re-scaled to reflectance range, of regions higher than cf threshold
        image: ee.Image(random_inputated.select(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']).first()),
        description: date_label+'_'+modis_coll+'_background_removal_random_inputated_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        folder: gDriveFolder,
        fileNamePrefix: date_label+'_'+modis_coll+'_background_removal_random_inputated_image_with_cf_'+int_cf_threshold+'perc_'+'grid_size'+grid_size,
        scale: 500,
        maxPixels: 1e11,
        crs: 'EPSG:4326',
        region: global_region
    }
);

Export.table.toDrive({
    // Export statistics grid 
    collection: ee.FeatureCollection(combined),
    description:date_label+'_'+modis_coll+'_stats_'+'grid_size'+grid_size,
    folder: gDriveFolder,
    fileNamePrefix: date_label+'_'+modis_coll+'_stats_'+'grid_size'+grid_size,
    fileFormat: 'shp'
});