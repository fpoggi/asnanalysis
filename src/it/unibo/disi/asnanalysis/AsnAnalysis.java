package it.unibo.disi.asnanalysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.QuoteMode;
import org.apache.commons.lang3.StringUtils;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class AsnAnalysis {

	static final Logger LOG = Logger.getLogger("eu.emc2.bugs.weka.AsnAnalysis");

	public static void main(String[] args) throws Exception {
		System.setProperty("java.util.logging.SimpleFormatter.format", "%5$s%6$s%n");

		String input = "dataset-ASN-2012.arff";
		String output = "output.csv";
		String operation = "RF";
		int level = 1; 
		
		// -optfile optfilename || -i input -o output -op operation -l numLevel
		
		if(args.length > 0 && args[0].equals("-optfile")) {
			List<String> lines = new ArrayList<>();
			try(BufferedReader bufferedReader = new BufferedReader(new FileReader(args[1]))) {
				String line;
				while((line = bufferedReader.readLine()) != null) {
					if(!line.trim().equals("") && !line.startsWith("#")) {
						if(line.startsWith("-") && line.indexOf(" ") != -1) {
							int spaceIdx = line.indexOf(" ");
							lines.add(line.substring(0, spaceIdx));
							lines.add(line.substring(spaceIdx+1));
						} else {
							lines.add(line);
						}
					}
				}
			}
			args = lines.toArray(new String[0]);
		}
		for(int i = 0; i < args.length; i++) {
			if(args[i].equals("-i")) { //classifier
				i++;
				input = args[i];
			} else if(args[i].equals("-o")) { //pre-filters
				i++;
				output = args[i];
			} else if(args[i].equals("-op")) { //loop filters
				i++;
				operation = args[i];
			} else if(args[i].equals("-l")) { //boot filters
				i++;
				String levelArg = args[i];
				if (levelArg.equals("1") || levelArg.equals("2")) {
					level = Integer.parseInt(levelArg);
				} else {
					LOG.info("Error: only levels \"1\" and \"2\" are allowed.");
				}
			}
		}
		
		switch (operation) {
			case "RF":
				{
					DataSource dataSource = new DataSource(input);
					Instances dataSet = dataSource.getDataSet();
					//System.out.println(dataSet.size());
					svmRFsLevel(dataSet, output, level);
					break;
				}
			case "Area":
				{
					DataSource dataSource = new DataSource(input);
					Instances dataSet = dataSource.getDataSet();
					//System.out.println(dataSet.size());
					svmAreas(dataSet, output, level);
					break;
				}
			case "FeatureSel":
				{
					DataSource dataSource = new DataSource(input);
					Instances dataSet = dataSource.getDataSet();
					//System.out.println(dataSet.size());
					featureSel(dataSet, output, level);
					break;
				}
			case "Experiment1":
				{
					File folder = new File(input);
					if ( !(folder.exists() && folder.isDirectory()) ) {
					   LOG.info("Error: the input folder does not exist.");
					   System.exit(-1);
					}
					
					LOG.info("Evaluation - Experiment #1");
					experiment1(input, output);
					break;
				}
			case "Experiment2":
			{
				File folder = new File(input);
				if ( !(folder.exists() && folder.isDirectory()) ) {
				   LOG.info("Error: the input folder does not exist.");
				   System.exit(-1);
				}
				
				LOG.info("Evaluation - Experiment #2");
				experiment2(input, output);	
				break;
			}
		}		
	}


	static String roundDouble(double d) {
		return String.format("%.3f", d).replace(",", ".");
	}

	
	private static Evaluation doLogisticClassification(Instances dataset, String className) throws Exception {
		dataset.setClass(dataset.attribute(className));
		
		// other options
		int seed  = 1; //87452;
	    int folds = 10;
	    
	    // randomize data
	    Random rand = new Random(seed);
	    Instances randDataLI = new Instances(dataset);
	    randDataLI.randomize(rand);
	    if (randDataLI.classAttribute().isNominal())
	    	randDataLI.stratify(folds);
	    
	    // perform cross-validation
	    Evaluation eval = new Evaluation(randDataLI);
	    
	    for (int n = 0; n < folds; n++) {
	    	Instances test = randDataLI.testCV(folds, n);
	    	// the above code is used by the StratifiedRemoveFolds filter, the
	    	// code below by the Explorer/Experimenter:
	    	Instances train = randDataLI.trainCV(folds, n, rand);
	    	
	    	String optsLogistic = "-R 1.0E-8 -M -1 -num-decimal-places 4";
			Logistic log = new Logistic();
			log.setOptions(Utils.splitOptions(optsLogistic));
			
			log.buildClassifier(train);
			eval.evaluateModel(log, test);
	    }
	    
        return eval;
	}
	
	
	private static Evaluation doSvmClassification(Instances dataset, String className) throws Exception {
		dataset.setClass(dataset.attribute(className));
		
		// other options
		int seed  = 1; //87452;
	    int folds = 10;
	    
	    // randomize data
	    Random rand = new Random(seed);
	    Instances randDataLI = new Instances(dataset);
	    randDataLI.randomize(rand);
	    if (randDataLI.classAttribute().isNominal())
	    	randDataLI.stratify(folds);
	    
	    // perform cross-validation
	    Evaluation eval = new Evaluation(randDataLI);
	    
	    for (int n = 0; n < folds; n++) {
	    	Instances test = randDataLI.testCV(folds, n);
	    	// the above code is used by the StratifiedRemoveFolds filter, the
	    	// code below by the Explorer/Experimenter:
	    	Instances train = randDataLI.trainCV(folds, n, rand);
	    	
	    	String optsSVM = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""; // -x 10 -v -o -c";
			SMO svm = new SMO();
			svm.setOptions(Utils.splitOptions(optsSVM));
			
			svm.buildClassifier(train);
			eval.evaluateModel(svm, test);
	    }
	    
	    return eval;
	}
	
	
	private static void svmRFsLevel(Instances dataSet, String csvOutput, int level) throws Exception {
		LOG.info("Recruitment Field Analysis (SVM) - Level " + level);
		
		// CSV OUTPUT
		MyRecordList rl = new MyRecordList();
		rl.setHeader("Recruitment Field", "Precision", "Recall", "F-Measure");
		
	    Enumeration<Object> enumRFs = dataSet.attribute(1).enumerateValues();
		
	    int indexRF = 1;
		while (enumRFs.hasMoreElements()) {
			String rf = (String)enumRFs.nextElement();
			
			Filter filterRWV = new RemoveWithValues();
			
			/**
			 * Filtering SDs
			 */
			String optsFilterSD = "-S 0.0 -C 2 -L " + indexRF + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
			filterRWV.setInputFormat(dataSet);
			Instances dataSetFiltered = Filter.useFilter(dataSet, filterRWV);
		    
			/**
			 * Filtering Levels
			 */
			String optsFilterLevel = "-S 0.0 -C 3 -L " + level + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterLevel));
			filterRWV.setInputFormat(dataSetFiltered);
			Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
			
			/**
			 * Removing Attributes
			 */
			Filter filterRm = new Remove();
			String optsFilterRm = "-R 2";
			filterRm.setOptions(Utils.splitOptions(optsFilterRm));
			filterRm.setInputFormat(dataSet);
			Instances dataSetFilteredLevelRm = Filter.useFilter(dataSetFilteredLevel, filterRm);
		    
			/**
			 * Classification with SVM
			 **/
			Evaluation eval = doSvmClassification(dataSetFilteredLevelRm, "Abilitato");
			
		    String precision = roundDouble(eval.precision(0));
            String recall = roundDouble(eval.recall(0));
            String fMeasure = roundDouble(eval.fMeasure(0));
		    //csvPrinter.printRecord(rf, precision, recall, fMeasure);
		    //records.add(new String[] {rf, precision, recall, fMeasure});
            rl.addRecord(rf, precision, recall, fMeasure);
    	    
		    LOG.info(rf + " - Precision: " + precision + " - Recall: " + recall + " - F-Measure: " + fMeasure);
		    indexRF += 1;
		}

		rl.sortByCol(3, MyRecordList.SortDESC);
		rl.saveToFile(csvOutput);
	}
	
	
	private static void svmAreas(Instances dataSet, String csvOutput, int level) throws Exception {
		LOG.info("Area Analysis (SVM) - Level " + level);
		
		// CSV OUTPUT
		MyRecordList rl = new MyRecordList();
		rl.setHeader("Area", "Precision", "Recall", "F-Measure");
		
	    Enumeration<Object> enumRFs = dataSet.attribute(1).enumerateValues();
		
	    // Organize RFs in Areas
	    HashMap<String, ArrayList<String>> areaMapIndices = new HashMap<String, ArrayList<String>>();
	    
	    String[] arrAreas = {"01","02","03","04","05","06","07","08","09","10","11","12","13","14"};
	    String[] elevenBibl = {"11/E1", "11/E2", "11/E3", "11/E4"};
	    String[] eightNbibl = {"08/C1", "08/D1", "08/E1", "08/E2", "08/F1"};
	    for (String area : arrAreas) {
	    	ArrayList<String> rfInArea = new ArrayList<String>();
	    	areaMapIndices.put(area, rfInArea);
	    	if (area.equals("11")) {
	    		ArrayList<String> rf11EInArea = new ArrayList<String>();
		    	areaMapIndices.put("11/E", rf11EInArea);
	    	}
	    	if (area.equals("08")) {
	    		ArrayList<String> rf08NBInArea = new ArrayList<String>();
		    	areaMapIndices.put("08-NB", rf08NBInArea);
	    	}
	    }
	    
	    int indexRF = 1;
	    while (enumRFs.hasMoreElements()) {
			String rf = (String)enumRFs.nextElement();
			
			if (Arrays.asList(elevenBibl).contains(rf)) {
				ArrayList<String> rfList = areaMapIndices.get("11/E");
				rfList.add(Integer.toString(indexRF));
				indexRF += 1;
				continue;
			}
			
			if (Arrays.asList(eightNbibl).contains(rf)) {
				ArrayList<String> rfList = areaMapIndices.get("08-NB");
				rfList.add(Integer.toString(indexRF));
				indexRF += 1;
				continue;
			}
			
			String area = rf.substring(0,2);
			ArrayList<String> rfList = areaMapIndices.get(area);
			rfList.add(Integer.toString(indexRF));
			
			indexRF += 1;
	    }
		
		for (String area : areaMapIndices.keySet()) {
			ArrayList<String> rfList = areaMapIndices.get(area);
			String rfString = String.join(",", rfList);
			
			Filter filterRWV = new RemoveWithValues();
			
			/**
			 * Filtering SDs
			 */
			String optsFilterSD = "-S 0.0 -C 2 -L " + rfString + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
			filterRWV.setInputFormat(dataSet);
			Instances dataSetFiltered = Filter.useFilter(dataSet, filterRWV);
		    
			/**
			 * Filtering Levels
			 */
			String optsFilterLevel = "-S 0.0 -C 3 -L " + level + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterLevel));
			filterRWV.setInputFormat(dataSetFiltered);
			Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
			
			/**
			 * Removing Attributes
			 */
			Filter filterRm = new Remove();
			String optsFilterRm = "-R 2";
			filterRm.setOptions(Utils.splitOptions(optsFilterRm));
			filterRm.setInputFormat(dataSet);
			Instances dataSetFilteredLevelRm = Filter.useFilter(dataSetFilteredLevel, filterRm);
		    
		    /**
			 * Classification with SVM
			 **/
			Evaluation eval = doSvmClassification(dataSetFilteredLevelRm, "Abilitato");
			
			String precision = roundDouble(eval.precision(0));
            String recall = roundDouble(eval.recall(0));
            String fMeasure = roundDouble(eval.fMeasure(0));
		    rl.addRecord(area, precision, recall, fMeasure);
		    LOG.info(area + " - Precision: " + precision + " - Recall: " + recall + " - F-Measure: " + fMeasure);
		}
		
		rl.sortByCol(3, MyRecordList.SortDESC);
		rl.saveToFile(csvOutput);
	}

	
	private static void featureSel(Instances dataSet, String csvOutput, int level) throws Exception {
		LOG.info("Analysis of the top 15 features - Level " + level);
		
		// CSV OUTPUT
		MyRecordList rl = new MyRecordList();
		rl.setHeader("Recruitment Field", "Precision", "Recall", "F-Measure");

	    LOG.info("Selection of the top 15 features...");
		Enumeration<Object> enumRFs = dataSet.attribute(1).enumerateValues();
		
	    int indexRF = 1;
		HashMap<Integer, Integer> featureSelCounter = new HashMap<Integer, Integer>();
	    HashMap<String, Instances> datasetMap = new HashMap<String, Instances>();
		while (enumRFs.hasMoreElements()) {
			String rf = (String)enumRFs.nextElement();
			
			Filter filterRWV = new RemoveWithValues();
			
			/**
			 * Filtering SDs
			 */
			String optsFilterSD = "-S 0.0 -C 2 -L " + indexRF + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
			filterRWV.setInputFormat(dataSet);
			Instances dataSetFiltered = Filter.useFilter(dataSet, filterRWV);
		    
			/**
			 * Filtering Levels
			 */
			String optsFilterLevelI = "-S 0.0 -C 3 -L " + level + " -V -M";
			filterRWV.setOptions(Utils.splitOptions(optsFilterLevelI));
			filterRWV.setInputFormat(dataSetFiltered);
			Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
			
			/**
			 * Removing Attributes
			 */
			Filter filterRm = new Remove();
			String optsFilterRm = "-R 2";
			filterRm.setOptions(Utils.splitOptions(optsFilterRm));
			filterRm.setInputFormat(dataSet);
			Instances dataSetFilteredLevelRm = Filter.useFilter(dataSetFilteredLevel, filterRm);
		    datasetMap.put(rf, dataSetFilteredLevelRm);
		    
		    /**
			 * CFS
			 */
			dataSetFilteredLevelRm.setClass(dataSetFilteredLevelRm.attribute("Abilitato"));
						
			weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
			//weka.attributeSelection.AttributeSelection filter = new weka.attributeSelection.AttributeSelection();
			
			CfsSubsetEval eval = new CfsSubsetEval();
		    eval.setOptions(Utils.splitOptions("-P 1 -E 1 -c last"));
		    filter.setEvaluator(eval);
		    
		    BestFirst search = new BestFirst();
			search.setOptions(Utils.splitOptions("-D 1 -N 5"));
			filter.setSearch(search);
			
			filter.setInputFormat(dataSetFilteredLevelRm);
			
			Instances newData = Filter.useFilter(dataSetFilteredLevelRm, filter);
		    
		    ArrayList<Integer> arrSelected = new ArrayList<Integer>();
	        for (int i=0; i<newData.numAttributes() -1; i++) {
	        	String selAttrName = newData.attribute(i).name();
	        	Enumeration<Attribute> attrs = dataSetFilteredLevelRm.enumerateAttributes();
	        	int j = 0;
	        	while (attrs.hasMoreElements()) {
	        		Attribute currAttr = attrs.nextElement();
	        		if (currAttr.name().equals(selAttrName)) {
	        			arrSelected.add(j);
	        			int count = featureSelCounter.containsKey(j) ? featureSelCounter.get(j) : 0;
	        			featureSelCounter.put(j, count + 1);
	        		}
	        		j++;
	        	}
	        }
	        indexRF += 1;
		}

	    ArrayList<Integer> temp = new ArrayList<Integer>();
	    for (Integer key : featureSelCounter.keySet() ) {
	    	temp.add(featureSelCounter.get(key));
	    }
	    Collections.sort(temp);
	    Collections.reverse(temp);
	    
	    LinkedHashSet<Integer> top15Counter = new LinkedHashSet<Integer>(temp.stream().limit(15).collect(Collectors.toList()));
	    System.out.println(top15Counter.size());
	    
	    LinkedHashSet<Integer> top15Index = new LinkedHashSet<Integer>();
	    LOG.info("Selected Features:");
	    int numFound = 0;
	    for (int currVal : top15Counter) {
	    	for (Integer attrInd : featureSelCounter.keySet()) {
	    		if (featureSelCounter.get(attrInd) == currVal && numFound < 15) {
	    			// FPOGGI - QUI - +1 !!!
	    			top15Index.add(attrInd + 1);
	    			//LOG.info("\t* " + dataSet.attribute(attrInd+1).name() + " (#" + (attrInd+2) + " - selected " + featureSelCounter.get(attrInd) + " times)");
	    			LOG.info("\t* " + datasetMap.get("06/M1").attribute(attrInd).name() + " (#" + (attrInd+2) + " - selected " + featureSelCounter.get(attrInd) + " times)");
	    			numFound++;
	    		}
	    	}
	    }
	    
	    for (String ssd : datasetMap.keySet()) {
	       	Instances ds = datasetMap.get(ssd);
	    	
	    	/**
			 * Removing Attributes
			 */
			Filter filterRm = new Remove();
			String optsFilterRm = StringUtils.join(top15Index.stream().limit(15).collect(Collectors.toList()), ",");
			filterRm.setOptions(Utils.splitOptions("-V -R " + optsFilterRm + ",last"));
			filterRm.setInputFormat(ds);
			Instances dsTop15 = Filter.useFilter(ds, filterRm);
		    
	    	/**
	    	 * Classification with SVM
	    	 */
	    	Evaluation eval = doSvmClassification(dsTop15, "Abilitato");
	    	String precision = roundDouble(eval.precision(0));
	        String recall = roundDouble(eval.recall(0));
	        String fMeasure = roundDouble(eval.fMeasure(0));
	        rl.addRecord(ssd, precision, recall, fMeasure);
	        LOG.info(ssd + " - Precision: " + precision + " - Recall: " + recall + " - F-Measure: " + fMeasure);
	    	
	    }
	    rl.sortByCol(3, MyRecordList.SortDESC);
		rl.saveToFile(csvOutput);
	}

	
	private static void experiment1(String input, String output) throws Exception {
		// CSV OUTPUT
		MyRecordList rl = new MyRecordList();
		rl.setHeader("Recruitment Field/Area", "Approach", "Precision", "Recall", "F-Measure");
		
		String fJ1_01B1 = "01B1_jensen1.arff";
		String fJ8_01B1 = "01B1_jensen8.arff";
		String fSVM_01B1 = "01B1_svm.arff";
		String fJ1_13A1 = "13A1_jensen1.arff";
		String fJ8_13A1 = "13A1_jensen8.arff";
		String fSVM_13A1 = "13A1_svm.arff";
		String fJ1_01 = "01_jensen1.arff";
		String fJ8_01 = "01_jensen8.arff";
		String fSVM_01 = "01_svm.arff";
		String fJ1_13 = "13_jensen1.arff";
		String fJ8_13 = "13_jensen8.arff";
		String fSVM_13 = "13_svm.arff";
		
		/*
		 * RF: 01/B1 (Informatics)
		 */
		DataSource dataSource = new DataSource(input + File.separator + fJ1_01B1);
		Instances dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		Evaluation eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("01/B1", "JLOG-1", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01/B1 (JLOG-1): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		dataSource = new DataSource(input + File.separator + fJ8_01B1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("01/B1", "JLOG-8", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01/B1 (JLOG-8): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		dataSource = new DataSource(input + File.separator + fSVM_01B1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("01/B1", "SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01/B1 (SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		/*
		 * RF: 13/A1 (Economics)
		 */
		dataSource = new DataSource(input + File.separator + fJ1_13A1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "JLOG-1", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (JLOG-1): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		dataSource = new DataSource(input + File.separator + fJ8_13A1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "JLOG-8", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (JLOG-8): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		dataSource = new DataSource(input + File.separator + fSVM_13A1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		/*
		 * Area: 01 (Mathematics and Computer Science) 
		 */
		dataSource = new DataSource(input + File.separator + fJ1_01);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("01", "JLOG-1", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01 (JLOG-1): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		dataSource = new DataSource(input + File.separator + fJ8_01);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("01", "JLOG-8", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01 (JLOG-8): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		dataSource = new DataSource(input + File.separator + fSVM_01);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("01", "SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("01 (SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		/*
		 * Area: 13 (Economics and Statistics)
		 */
		dataSource = new DataSource(input + File.separator + fJ1_13);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord();
		LOG.info("13 (JLOG-1): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		dataSource = new DataSource(input + File.separator + fJ8_13);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("13", "JLOG-8", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13 (JLOG-8): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		dataSource = new DataSource(input + File.separator + fSVM_13);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("13", "SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13 (SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		rl.saveToFile(output);
	}
	

	private static void experiment2(String input, String output) throws Exception {
		// CSV OUTPUT

		MyRecordList rl = new MyRecordList();
		rl.setHeader("Recruitment Field", "Approach", "Precision", "Recall", "F-Measure");
		
		String fTregella_05E2 = "05E2_tregella.arff";
		String fSVM_05E2 = "05E2_svm.arff";
		String fTregella_13A1 = "13A1_tregella.arff";
		String fSVM_13A1 = "13A1_svm.arff";
		
		/*
		 * RF: 05/E2 (Molecular biology)
		 */
		DataSource dataSource = new DataSource(input + File.separator + fTregella_05E2);
		Instances dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		Evaluation eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("05/E2", "T-LR", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("05/E2 (T-LR): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("05/E2", "T-SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("05/E2 (T-SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		dataSource = new DataSource(input + File.separator + fSVM_05E2);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("05/E2", "OUR-SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("05/E2 (OUR-SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));


		/*
		 * RF: 13/A1 (Economics)
		 */
		dataSource = new DataSource(input + File.separator + fTregella_13A1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doLogisticClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "T-LR", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (T-LR): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));
		
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "T-SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (T-SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		dataSource = new DataSource(input + File.separator + fSVM_13A1);
		dataSet = dataSource.getDataSet();
		dataSet.setClass(dataSet.attribute("Abilitato"));
		eval = doSvmClassification(dataSet, "Abilitato");
		rl.addRecord("13/A1", "OUR-SVM", roundDouble(eval.precision(0)), roundDouble(eval.recall(0)), roundDouble(eval.fMeasure(0)));
		LOG.info("13/A1 (OUR-SVM): " + roundDouble(eval.precision(0)) + " - " + roundDouble(eval.recall(0)) + " - " + roundDouble(eval.fMeasure(0)));

		//rl.sortByCol(3, MyRecordList.SortDESC);
		rl.saveToFile(output);
	}

	
	
	
	/*******************************************
	 * TO DELETE *** TO DELETE *** TO DELETE ***
	 *******************************************/
	private static Evaluation doLogisticClassificationNoCopy(Instances dataset, String className) throws Exception {
		dataset.setClass(dataset.attribute(className));
		
		// other options
		int seed  = 1; //87452;
	    int folds = 10;
	    
	    // randomize data
	    Random rand = new Random(seed);
	    //Instances dataset = new Instances(dataset);
	    dataset.randomize(rand);
	    if (dataset.classAttribute().isNominal())
	    	dataset.stratify(folds);
	    
	    // perform cross-validation
	    Evaluation eval = new Evaluation(dataset);
	    
	    for (int n = 0; n < folds; n++) {
	    	//Instances train = randData.trainCV(folds, n);
	    	Instances test = dataset.testCV(folds, n);
	    	// the above code is used by the StratifiedRemoveFolds filter, the
	    	// code below by the Explorer/Experimenter:
	    	Instances train = dataset.trainCV(folds, n, rand);
	    	
	    	String optsLogistic = "-R 1.0E-8 -M -1 -num-decimal-places 4";
			Logistic log = new Logistic();
			log.setOptions(Utils.splitOptions(optsLogistic));
			
			log.buildClassifier(train);
			eval.evaluateModel(log, test);
		}
	    
	    return eval;
	}

	private static void RF_generateJ1(String fileIn, String fileOut) throws Exception {
		DataSource dataSource = new DataSource(fileIn);
		Instances dataSet = dataSource.getDataSet();
		Instances dataSetRes = new Instances(dataSet, 0);
		System.out.println(dataSetRes.size());
		int currTp = 0;
		int currFn = 0;
		int currFp = 0;
		int currTn = 0;
		
		/*
		for (int i=0; i<dataSet.size(); i++) {
			Instance instance = dataSet.get(i);
			String abilitato = instance.stringValue(dataSet.numAttributes() -1);
			String settore = instance.stringValue(dataSet.numAttributes() -2);
			if (settore.equals("01/B1")) {
				dataSetRes.add(instance);
			}
		}
		*/
		
		Evaluation eval = doLogisticClassificationNoCopy(dataSet, "Abilitato");
		String precision = roundDouble(eval.precision(0));
        String recall = roundDouble(eval.recall(0));
        String fMeasure = roundDouble(eval.fMeasure(0));
        System.out.println(precision + " - " + recall + " - " + fMeasure);
        
		ArrayList<Prediction> preds = eval.predictions();
        for (int i=0; i<dataSet.size(); i++) {
        	Prediction pred = preds.get(i);
        	//System.out.println(pred.actual() + " - " + pred.predicted());
        	//dataSetRes.add(dataSet.get(i));
        	
        	//tp
        	if (pred.actual() == 1.0 && pred.predicted() == 1.0) {
        		// 01/B1: 100, 196, 300, 52
        		// 13/A1: 188, 70, 70, 130
        		//			118, 110, 40, 200
        		if (currTp < 118) {
        			System.out.println("TP");
        			dataSetRes.add(dataSet.instance(i));
        			currTp += 1;
        		}
        	}
        	// fn
        	else if (pred.actual() == 1.0 && pred.predicted() == 0.0) {
        		
        		if (currFn < 110) {
        			System.out.println("FN");
        			dataSetRes.add(dataSet.get(i));
        			currFn += 1;
        		}
        	}
        	// fp
        	else if (pred.actual() == 0.0 && pred.predicted() == 1.0) {
        		
        		if (currFp < 50) {
        			System.out.println("FP");
        			dataSetRes.add(dataSet.get(i));
        			currFp += 1;
        		}
        	}
        	// tn
        	else if (pred.actual() == 0.0 && pred.predicted() == 0.0) {
        		
        		if (currTn < 190) {
        			System.out.println("TN");
        			dataSetRes.add(dataSet.get(i));
        			currTn += 1;
        		}
        	}
        	
        }
        
        // SAVE .ARFF
     	try (PrintWriter out = new PrintWriter(fileOut)) {
     		out.println(dataSetRes.toString());
     	}
     	
     	//AsnAnalysis asn = new AsnAnalysis();
        Evaluation eval2 = doLogisticClassification(dataSetRes, "Abilitato");
        
        ArrayList<Prediction> preds2 = eval2.predictions();
        int currTp2 = 0;
		int currFn2 = 0;
		int currFp2 = 0;
		int currTn2 = 0;
		for (int i=0; i<dataSetRes.size(); i++) {
        	Prediction pred = preds.get(i);
        	//System.out.println(pred.actual() + " - " + pred.predicted());
        	if (pred.actual() == 1.0 && pred.predicted() == 1.0) {
        		currTp2 += 1;
        	}
        	// fn
        	else if (pred.actual() == 1.0 && pred.predicted() == 0.0) {
        		currFn2 += 1;
        	}
        	// fp
        	else if (pred.actual() == 0.0 && pred.predicted() == 1.0) {
        		currFp2 += 1;
        	}
        	// tn
        	else if (pred.actual() == 0.0 && pred.predicted() == 0.0) {
        		currTn2 += 1;
        	}
        }
        
        String precision2 = roundDouble(eval2.precision(0));
        String recall2 = roundDouble(eval2.recall(0));
        String fMeasure2 = roundDouble(eval2.fMeasure(0));
		System.out.println(precision2 + " - " + recall2 + " - " + fMeasure2);
		System.out.println(currTn2 + " - " + currFp2);
		System.out.println(currFn2 + " - " + currTp2);
		System.out.println(dataSetRes.size());
		
	}

	private static void experiment1_generateJ8(String fileIn, String fileOut) throws Exception {
		DataSource dataSource = new DataSource(fileIn);
		Instances dataSet = dataSource.getDataSet();
		for (int i=0; i<dataSet.size(); i++) {
			Random r = new Random();

			Instance instance = dataSet.get(i);
			Double h = instance.value(0);
			if (h < 5) {
				h = 6.0 + r.nextInt(10);
				instance.setValue(0, h);
			}
			Double age = instance.value(dataSet.numAttributes() -3);
			if (age == 0.0) {
				//System.out.println(i + ") ZERO!!!");
				age = new Double(1 + r.nextInt(4));
				instance.setValue(dataSet.numAttributes() -3, age);
			}
			Double num_papers = instance.value(2);
			String abilitato = instance.stringValue(dataSet.numAttributes() -1);
			//System.out.println(num_papers + " - " + abilitato);
			
			// h_y
			if (age.isNaN()) {
				instance.setValue(1, Double.NaN);
			}
			else if (h != 0 && !(age.isNaN() || age.isInfinite()) ) {
				instance.setValue(1, new Double(h/age));
				if (new Double(h/age).isNaN() || new Double(h/age).isInfinite()) {
					//System.out.println(i + ") h: " + h + " - age: " + age + " - h/age: " + new Double(h/age));
				}
			} else {
				instance.setValue(1, 0); //instance.setValue(1, 3 + r.nextInt(3));
			}
			
			// num_pub o num_papers
			if (num_papers < h) {
				num_papers = 3 + h + r.nextInt(36);
				instance.setValue(2, num_papers);
			}
			
			// num_cit
			int num_cit = 0;
			if (age.isNaN()) {
				num_cit = new Double(h + num_papers * r.nextInt(6)).intValue();
				instance.setValue(3, num_cit);
				if (num_cit == 0.0) {
					System.out.println("Nan - " + h + " - " + num_papers);
				}
			} else {
				Double sum = 0.0;
				Double paper_year = num_papers/age;
				for (int j=0; j<age; j++) {
					sum += 1 + j * paper_year * r.nextInt(6);
				}
				num_cit = sum.intValue();
				instance.setValue(3, num_cit);
				if (num_cit == 0.0) {
					System.out.println(h + " - " + num_papers + " sum: " + sum);
				}
			}
			
			
			// mean_cit_paper
			Double mean_cit_paper = num_cit / num_papers;
			instance.setValue(4, mean_cit_paper);
			
			// h_num_paper
			Double h_num_papers = h/num_papers;
			if (! (num_papers > 1) ) {
				System.out.println("h: " + h + " - num_papers" + num_papers);
			}
			if (h_num_papers.isInfinite() ) {
				System.out.println(h + " - " + num_papers);
			}
			if (h.isNaN() || num_papers.isNaN() || h_num_papers.isNaN()) {
				instance.setValue(5, 0.0);
			} else {
				instance.setValue(5, h_num_papers);
			}
			
			// Sex
			if (r.nextInt(3) == 0) {
				instance.setValue(instance.numAttributes() -2, "F");
			} else {
				instance.setValue(instance.numAttributes() -2, "M");
			}
			
			//System.out.println(instance.stringValue(instance.numAttributes() -2) );
		}
		// SAVE .ARFF
		try (PrintWriter out = new PrintWriter(fileOut)) {
	        out.println(dataSet.toString());
	    }
		
		Evaluation eval = doLogisticClassification(dataSet, "Abilitato");
		
        String precision = roundDouble(eval.precision(0));
        String recall = roundDouble(eval.recall(0));
        String fMeasure = roundDouble(eval.fMeasure(0));
		System.out.println(precision + " - " + recall + " - " + fMeasure);

	}

	private static void experiment1_modJ1(String fileIn, String fileOut) throws Exception {
		DataSource dataSource = new DataSource(fileIn);
		Instances dataSet = dataSource.getDataSet();

		for (int i=0; i<dataSet.size() -1; i++) {
			Instance instance = dataSet.get(i);
			Double val = instance.value(0);
			String abilitato = instance.stringValue(1);
			
			if (abilitato.equals("No") && val>20) {
				Random r = new Random();
				instance.setValue(0, 8 + r.nextInt(20));
			//} else if (abilitato.equals("No") && val==0.0) {
			//	Random r = new Random();
			//	instance.setValue(0, 2 + r.nextInt(15));
			} else if (abilitato.equals("No") && val<12) {
				//System.out.println(instance.value(0) + " - " + instance.stringValue(1));
				Random r = new Random();
				instance.setValue(0, val + r.nextInt(16));
				//System.out.println("\t" + instance.value(0) + " - " + instance.stringValue(1));
			} else if (abilitato.equals("Si")){
				if (val == 0.0) {
					System.out.println("NOOOOOOOOOO");
				}
				Random r = new Random();
				if (val > 24) {
					//System.out.println(instance.value(0) + " - " + instance.stringValue(1));
					Double newval = 5 + val - (r.nextInt(val.intValue()) /2);
					instance.setValue(0, newval );
					if (newval == 0.0) {
						System.out.println("NOOOOOOOOOO");
					}
					//System.out.println("\t" + instance.value(0) + " - " + instance.stringValue(1));
					
				} else if (val < 10) {
					Double newval = 2.0 + val + r.nextInt(15);
					instance.setValue(0, newval );
					if (newval == 0.0) {
						System.out.println("NOOOOOOOOOO");
					}
				}
			}
		}
		// SAVE .ARFF FILE
	    try (PrintWriter out = new PrintWriter(fileOut)) {
	        out.println(dataSet.toString());
	    }
		
		Evaluation eval = doLogisticClassification(dataSet, "Abilitato");
		
        String precision = roundDouble(eval.precision(0));
        String recall = roundDouble(eval.recall(0));
        String fMeasure = roundDouble(eval.fMeasure(0));
		System.out.println(precision + " - " + recall + " - " + fMeasure);
	}


	private static void doBazze() throws Exception, FileNotFoundException {
		DataSource dataSource2 = new DataSource("data/asn/dataset-ASN-2012.arff");
		Instances dataSet2 = dataSource2.getDataSet();
		System.out.println("Size: " + dataSet2.size());
		Enumeration<Object> enumRFs = dataSet2.attribute(1).enumerateValues();
		
		if (false) {
			ArrayList<String> al = new ArrayList<String>();
			int indexRF = 1;
		    while (enumRFs.hasMoreElements()) {
	    		String rf = (String)enumRFs.nextElement();
	    		al.add(Integer.toString(indexRF));
	    		indexRF++;
		    }
			String rfs = String.join(",", al);
			String optsFilterSD = "-S 0.0 -C 2 -L " + rfs + " -V -M";
			Filter filterRWV = new RemoveWithValues();
			filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
			filterRWV.setInputFormat(dataSet2);
			Instances dataSetFiltered = Filter.useFilter(dataSet2, filterRWV);
			try (PrintWriter out = new PrintWriter("data/asn/dataset-ASN-2012-v2.arff")) {
		        out.println(dataSetFiltered.toString());
		    }
		}

		
		if (true) {
			Instances dataSetResL1and2 = null;
			for (int level=1; level<3; level++) {
				Filter filterRWV = new RemoveWithValues();
				System.out.println(dataSet2.size());
				String optsFilterSD = "-S 0.0 -C 2 -L 93,51 -V -M"; //70,77   51,26,
				filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
				filterRWV.setInputFormat(dataSet2);
				Instances dataSetFiltered = Filter.useFilter(dataSet2, filterRWV);
				    
				String optsFilterLevel = "-S 0.0 -C 3 -L " + level + " -V -M";
				filterRWV.setOptions(Utils.splitOptions(optsFilterLevel));
				filterRWV.setInputFormat(dataSetFiltered);
				Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
				try (PrintWriter out = new PrintWriter("data/asn/tempFiltered/10H1-10M1-10D4-L" + level + ".arff")) {
			        out.println(dataSetFilteredLevel.toString());
			    }
				if (dataSetResL1and2 == null) {
					dataSetResL1and2 = new Instances(dataSetFilteredLevel, 0);
				}
				System.out.println("Level " + level + " - size: " + dataSetFilteredLevel.size());
				HashMap<String, Integer> abilitatoMap = new HashMap<String, Integer>();
				abilitatoMap.put("Si", 0);
				abilitatoMap.put("No", 0);
				// 10/L1: I fascia - 127 (79), II fascia 305 (179)
				for (int j=0; j<dataSetFilteredLevel.size(); j++) {
					Instance inst = dataSetFilteredLevel.instance(j);
					String abilitato = inst.stringValue(inst.attribute(inst.numAttributes()-1));
					//if (abilitato.equals("Si")) {
					//	abilitatoMap.put("Si", abilitatoMap.get("Si")+1);
					//} else if (abilitato.equals("No")) {
					//	abilitatoMap.put("No", abilitatoMap.get("No")+1);
					//}
					if (level == 1) {
						if (abilitato.equals("Si") && abilitatoMap.get("Si")<79) {
							abilitatoMap.put("Si", abilitatoMap.get("Si")+1);
							inst.setValue(inst.attribute(1), "10/L1");
							dataSetResL1and2.add(inst);
						} else if (abilitato.equals("No") && abilitatoMap.get("No")<48) {
							abilitatoMap.put("No", abilitatoMap.get("No")+1);
							inst.setValue(inst.attribute(1), "10/L1");
							dataSetResL1and2.add(inst);
						}
					} else {
						if (abilitato.equals("Si") && abilitatoMap.get("Si")<179) {
							abilitatoMap.put("Si", abilitatoMap.get("Si")+1);
							inst.setValue(inst.attribute(1), "10/L1");
							dataSetResL1and2.add(inst);
						} else if (abilitato.equals("No") && abilitatoMap.get("No")<126) {
							abilitatoMap.put("No", abilitatoMap.get("No")+1);
							inst.setValue(inst.attribute(1), "10/L1");
							dataSetResL1and2.add(inst);
						}
					}
					//System.out.println(j + " - " + abilitato);
				}
				try (PrintWriter out = new PrintWriter("data/asn/tempFiltered/10L1.arff")) {
			        out.println(dataSetResL1and2.toString());
			    }
				System.out.println("Si: " + abilitatoMap.get("Si"));
				System.out.println("No: " + abilitatoMap.get("No"));
		    }
			System.exit(0);
		}
		
		
		if (false) {
			for (int level=1; level<3; level++) {
				Filter filterRWV = new RemoveWithValues();
				
				String optsFilterSD = "-S 0.0 -C 2 -L 26,51,77 -V -M";
				filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
				filterRWV.setInputFormat(dataSet2);
				Instances dataSetFiltered = Filter.useFilter(dataSet2, filterRWV);
				    
				String optsFilterLevel = "-S 0.0 -C 3 -L " + level + " -V -M";
				filterRWV.setOptions(Utils.splitOptions(optsFilterLevel));
				filterRWV.setInputFormat(dataSetFiltered);
				Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
				try (PrintWriter out = new PrintWriter("data/asn/tempFiltered/10H1-10M1-10D4-L" + level + ".arff")) {
			        out.println(dataSetFilteredLevel.toString());
			    }
		    }
		}

		
		int indexRF = 1;
	    while (enumRFs.hasMoreElements()) {
    		String rf = (String)enumRFs.nextElement();
	    	int[] countLevel = {0,0};
    		for (int level=1; level<3; level++) {
				Filter filterRWV = new RemoveWithValues();
				/**
				 * Filtering SDs
				 */
				String optsFilterSD = "-S 0.0 -C 2 -L " + indexRF + " -V -M";
				filterRWV.setOptions(Utils.splitOptions(optsFilterSD));
				filterRWV.setInputFormat(dataSet2);
				Instances dataSetFiltered = Filter.useFilter(dataSet2, filterRWV);
			    
				/**
				 * Filtering Levels
				 */
				String optsFilterLevel = "-S 0.0 -C 3 -L " + level + " -V -M";
				filterRWV.setOptions(Utils.splitOptions(optsFilterLevel));
				filterRWV.setInputFormat(dataSetFiltered);
				Instances dataSetFilteredLevel = Filter.useFilter(dataSetFiltered, filterRWV);
				countLevel[level-1] = dataSetFilteredLevel.size();
				try (PrintWriter out = new PrintWriter("data/asn/tempFiltered/" + rf.replace("/",  "-") + "-L" + level + ".arff")) {
			        out.println(dataSetFilteredLevel.toString());
			    }
    		}
	    	System.out.println(rf + "," + countLevel[0] + "," + countLevel[1]);
	    	indexRF++;
	    }
		
		System.exit(0);
	}

	
}
