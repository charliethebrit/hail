package is.hail.methods

import is.hail.utils._
import java.io._

import scala.io.Source
import is.hail.expr._
import is.hail.SparkSuite
import is.hail.io.annotators.IntervalList
import is.hail.variant.{Genotype, VariantDataset}

import util.Random.shuffle
import scala.sys.process._
import breeze.linalg._

import scala.language.postfixOps
import is.hail.methods.Skat.keyedRDDSkat
import is.hail.stats.RegressionUtils.getSampleAnnotation
import is.hail.stats.RegressionUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

case class SkatAggForR[T <: Vector[Double]](xs: ArrayBuilder[T], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {

  def noiseTest() = {
    val noiseTests = 10
    val averageOver = 1
    val noiseIncrement =   .5
    val useDosages = false
    val useLargeN = false
    val useLogistic = true

    val fileLocation = "/Users/charlesc/Documents/Software/SkatExperiments/alternatingDemo.vds"
    val saveFileLocation = "/Users/charlesc/Documents/Software/SkatExperiments/pValResults/BDNoiseTests4.txt"
    val pvals = Array.fill[Double](noiseTests)(0.0)
    val pvalsR = Array.fill[Double](noiseTests)(0.0)
    var pvalAve = 0.0
    var pvalRAve = 0.0

    var vds = hc.readVDS(fileLocation).annotateVariantsExpr("va.weight = global.weight[v]**2")

    //normalize data

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))
    val ec = EvalContext(symTab)
    val yIS = getSampleAnnotation(vds, "sa.pheno", ec).map(_.get)
    val yMean = sum(yIS)/yIS.length
    val yStd = math.sqrt(sum(yIS.map((x) => math.pow((x - yMean),2))) / yIS.length)

    vds = vds.annotateGlobalExpr(f"global.mean = $yMean")
             .annotateGlobalExpr(f"global.std = $yStd ")
             .annotateSamplesExpr("sa.pheno = (sa.pheno - global.mean)/global.std")

    var i = 0
    var j = 0
    while (i < noiseTests){

      while(j < averageOver) {
        val expr = f"sa.pheno = pcoin((1/(1 + exp(-(sa.pheno + $i%f * $noiseIncrement%f * rnorm(0,1))))))"
        //val expr = f"sa.pheno = sa.pheno + $i%f * $noiseIncrement%f * rnorm(0,1)"
        vds = vds.annotateSamples(vds.sampleIds.zip(yIS).toMap, TFloat64, "sa.pheno")
                 .annotateSamplesExpr(expr)

        val row = Skat(vds,"gene", "\'bestgeneever\'", singleKey = true, Some("va.weight"), "sa.pheno",
                       Array("sa.cov1", "sa.cov2"), useDosages, useLargeN, useLogistic).rdd.collect()
        val resultsArray = testInR(vds, "gene", "\'bestgeneever\'", singleKey = true,
          Some("va.weight"), "sa.pheno", Array("sa.cov1", "sa.cov2"), useDosages, useLogistic)

        pvalAve += row(0)(2).asInstanceOf[Double]
        pvalRAve += resultsArray(0)(2).asInstanceOf[Double]

        j += 1
      }
      pvals(i) = pvalAve/averageOver
      pvalsR(i) = pvalRAve/averageOver
      println((pvals(i), pvalsR(i)))
      i += 1
      j = 0

      pvalAve = 0
      pvalRAve = 0

    }
  }

 def permutationTest() = {
    val permutationTests = 100
    val pvalFileLocation = "/Users/charlesc/Documents/Software/SkatExperiments/pvalResults/constantPvals.txt"
    val fileLocation = "/Users/charlesc/Documents/Software/SkatExperiments/constantDemo.vds"
    val pythonFileLocation = "/Users/charlesc/Documents/Software/SkatExperiments/comparePvals.py"

    val pvals = Array.fill[Double](permutationTests + 1)(0.0)
    val pvalsR = Array.fill[Double](permutationTests + 1)(0.0)
    val useDosages = false
    val useLargeN = false
    val useLogistic = false

    var i = 0

    var vds = hc.readVDS(fileLocation).annotateVariantsExpr("va.weight = global.weight[v]**2")
                .annotateSamplesExpr("sa.pheno = pcoin((1/(1 + exp(-sa.pheno + 50))))")

    val symTab = Map(
      "s" -> (0, TString),
      "sa" -> (1, vds.saSignature))
    val ec = EvalContext(symTab)
    val yIS = getSampleAnnotation(vds, "sa.pheno", ec).map(_.get)

    var row = Skat(vds,"gene", "\'bestgeneever\'", singleKey = true, Some("va.weight"), "sa.pheno",
                    Array("sa.cov1", "sa.cov2"), useDosages, useLargeN, useLogistic).rdd.collect()
    pvals(0) = row(0)(1).asInstanceOf[Double]
    var resultsArray = testInR(vds, "gene", "\'bestgeneever\'", singleKey = true,
      Some("va.weight"), "sa.pheno", Array("sa.cov1", "sa.cov2"), useDosages, useLogistic)
    pvalsR(0) = resultsArray(0)(1).asInstanceOf[Double]

    println(row(0)(2).asInstanceOf[Double],resultsArray(0)(2).asInstanceOf[Double])
    println((pvals(0), pvalsR(0)))

    while(i < permutationTests){
      vds = vds.annotateSamples(vds.sampleIds.zip(shuffle(yIS)).toMap, TFloat64, "sa.pheno")
      row = Skat(vds,"gene", "\'bestgeneever\'", singleKey = true, Some("va.weight"), "sa.pheno",
                 Array("sa.cov1", "sa.cov2"), false, true).rdd.collect()
      pvals(i+1) = row(0)(1).asInstanceOf[Double]

      resultsArray = testInR(vds, "gene", "\'bestgeneever\'", singleKey = true,
        Some("va.weight"), "sa.pheno", Array("sa.cov1", "sa.cov2"), false, true)
      pvalsR(i+1) = resultsArray(0)(1).asInstanceOf[Double]

      i += 1

    }
    val pvalMatrix = new DenseMatrix(permutationTests + 1, 1, pvals)
    val pvalRMatrix = new DenseMatrix(permutationTests + 1, 1, pvalsR)
    val sendToPython = DenseMatrix.horzcat(pvalMatrix, pvalRMatrix)


    hadoopConf.writeTextFile(pvalFileLocation) {
      _.write(largeMatrixToString(sendToPython, ","))
    }

    val pyScript = "Python " + pythonFileLocation + s" ${ pvalFileLocation }"
    pyScript !

    println("blah blah")
  }

  def plotPvals(pvals: DenseMatrix[Double]) = {

    val inputFile= tmpDir.createLocalTempFile("pValMatrix", ".txt")
    hadoopConf.writeTextFile(inputFile) {
      _.write(largeMatrixToString(pvals,","))
    }

    val pyScript = s"Python " +
      s"${ inputFile }"
    pyScript !

  }

  //Routines for running programs in R for comparison

  def readResults(file: String) = {
    hadoopConf.readLines(file) {
      _.map {
        _.map {
          _.split(" ").map(_.toDouble)
        }.value
      }.toArray
    }
  }

  def largeMatrixToString(A: DenseMatrix[Double], separator: String): String = {
    var string: String = ""
    for (i <- 0 until A.rows) {
      for (j <- 0 until A.cols) {
        if (j == A.cols - 1)
          string = string + A(i, j).toString()
        else
          string = string + A(i, j).toString() + separator
      }
      string = string + "\n"
    }
    string
  }

  def sparseResultOp(st: Array[(SparseVector[Double], Double)], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val m = st.length
    val xArray = Array.ofDim[Double](m * n)
    val wArray = Array.ofDim[Double](m)

    var i = 0
    while (i < m) {
      val (xw, w) = st(i)
      wArray(i) = w

      val index = xw.index
      val data = xw.data
      var j = 0
      while (j < index.length) {
        xArray(i * n + index(j)) = data(j)
        j += 1
      }
      i += 1
    }

    (new DenseMatrix(n, m, xArray), new DenseVector(wArray))

  }

  def denseResultOp(st: Array[(DenseVector[Double], Double)], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val m = st.length
    val xArray = Array.ofDim[Double](m * n)
    val wArray = Array.ofDim[Double](m)

    var i = 0
    while (i < m) {
      val (xw, w) = st(i)
      wArray(i) = w

      val data = xw.data
      var j = 0
      while (j < n) {
        xArray(i * n + j) = data(j)
        j += 1
      }
      i += 1
    }

    (new DenseMatrix(n, m, xArray), new DenseVector(wArray))
  }

  def testInR(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: Option[String],
    yExpr: String,
    covExpr: Array[String],
    useDosages: Boolean,
    useLogistic: Boolean = false): Array[Row] = {


    def skatTestInR[T <: Vector[Double]](keyedRdd:  RDD[(Any, Iterable[(T, Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double], keyName: String,
      resultOp: (Array[(T, Double)], Int) => (DenseMatrix[Double], DenseVector[Double])): Array[Row] = {

      val n = y.size
      val k = cov.cols
      val d = n - k

      if (d < 1)
        fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

      // fit null model
      val qr.QR(q, r) = qr.reduced.impl_reduced_DM_Double(cov)
      val beta = r \ (q.t * y)
      val res = y - cov * beta
      val sigmaSq = (res dot res) / d

      val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
      hadoopConf.writeTextFile(inputFilePheno) {
        _.write(largeMatrixToString(y.toDenseMatrix, " "))
      }

      val inputFileCov = tmpDir.createLocalTempFile("skatCovMatrix", ".txt")
      hadoopConf.writeTextFile(inputFileCov) {
        _.write(largeMatrixToString(cov, " "))
      }

      val skatRDD = keyedRdd.collect()
        .map {case (k, vs) =>
          val (xs, weights) = resultOp(vs.toArray, n)

          //write files to a location R script can read
          val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
          hadoopConf.writeTextFile(inputFileG) {
            _.write(largeMatrixToString(xs, " "))
          }

          val inputFileW = tmpDir.createLocalTempFile("skatWeightVec", ".txt")
          hadoopConf.writeTextFile(inputFileW) {
            _.write(largeMatrixToString(weights.toDenseMatrix, " "))
          }

          val resultsFile = tmpDir.createLocalTempFile("results", ".txt")

          val datatype = if (useLogistic) "D" else "C"

          val rScript = s"Rscript src/test/resources/skatTest.R " +
            s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
            s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
            s"${ uriPath(resultsFile) } " + datatype

          rScript !
          val results = readResults(resultsFile)

          Row(k, results(0)(0), results(0)(1))
        }
      skatRDD
    }

    if (!useDosages) {
      val (keyedRdd, keysType, y, cov) =
        keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, RegressionUtils.hardCalls(_, _))
       skatTestInR(keyedRdd, keysType, y, cov, keyName, sparseResultOp _)
    }
    else {
      val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)
      val (keyedRdd, keysType, y, cov) =
        keyedRDDSkat(vds, variantKeys, singleKey, weightExpr, yExpr, covExpr, dosages)
      skatTestInR(keyedRdd, keysType, y, cov, keyName, denseResultOp _)
    }
  }


  def covariates = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TFloat64, "Cov2" -> TFloat64)).keyBy("Sample")

  def phenotypes = hc.importTable("src/" +
    "test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

  def phenotypesD = hc.importTable("src/" +
    "test/resources/skat.phenoD",
    types = Map("Pheno" -> TFloat64)).keyBy("Sample")

  def intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")

  def vds: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsTable(intervals, root = "va.genes", product = true)
    .annotateSamplesTable(phenotypes, root = "sa.pheno")
    .annotateSamplesTable(covariates, root = "sa.cov")
    .annotateSamplesExpr("sa.pheno = if (sa.pheno == 1.0) false else if (sa.pheno == 2.0) true else NA: Boolean")
    .filterSamplesExpr("sa.pheno.isDefined() && sa.cov.Cov1.isDefined() && sa.cov.Cov2.isDefined()")
    .annotateVariantsExpr("va.AF = gs.callStats(g=> v).AF")
    .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in dbeta(af,1.0,25.0)**2")

  @Test def hardcallsSmallTest() {

    val useDosages = false

    val kt = vds.skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {

      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]
      if (pval <= 1 && pval >= 0) {

        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }

      i += 1
    }

  }

  @Test def dosagesSmallTest() {

    val useDosages = true

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val kt = vds.skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)


    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }

  }

  @Test def largeNSmallTest() {

    val useDosages = false
    val useLargeN = true

    val kt = Skat(vds, "gene", "va.genes", false, Option("va.weight"), "sa.pheno",
      Array("sa.cov.Cov1", "sa.cov.Cov2"),useDosages, useLargeN)

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]


      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }

  }


  //Generates random data
  def buildWeightValues(filepath: String): Unit = {

    //read in chrom:pos values
    val fileSource = Source.fromFile(filepath)
    val linesArray = fileSource.getLines.toArray
    fileSource.close()

    //write in randomized weights
    val fileObject = new PrintWriter(new File(filepath))

    for (i <- -1 until linesArray.size) {
      if (i == -1) {
        fileObject.write("Pos\tWeights\n")
      }
      else {
        val pos = linesArray(i)
        val randomWeight = scala.util.Random.nextDouble()
        fileObject.write(s"$pos\t$randomWeight\n")

      }
    }
    fileObject.close()
  }

  def buildCovariateMatrix(filepath: String, covariateCount: Int): Unit = {
    val fileObject = new PrintWriter(new File(filepath))

    val startIndex = 96
    val endIndex = 116

    for (i <- startIndex - 1 to endIndex) {

      if (i == startIndex - 1) {
        fileObject.write("Sample\t")
        for (j <- 1 to covariateCount) {
          if (j == covariateCount) {
            fileObject.write(s"Cov$j")
          }
          else {
            fileObject.write(s"Cov$j\t")
          }
        }
        fileObject.write("\n")
      }
      else {
        fileObject.write("HG%05d\t".format(i))
        for (j <- 1 to covariateCount) {
          if (j == covariateCount) {
            fileObject.write("%d".format(scala.util.Random.nextInt(25)))
          }
          else {
            fileObject.write("%d\t".format(scala.util.Random.nextInt(25)))
          }
        }
        fileObject.write("\n")
      }
    }

    fileObject.close()
  }

  //Dataset for big Test

  def covariatesSkat = hc.importTable("src/test/resources/skat.cov",
    impute = true).keyBy("Sample")

  def phenotypesSkat = hc.importTable("src/test/resources/skat.pheno",
    types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

  def intervalsSkat = IntervalList.read(hc, "src/test/resources/skat.interval_list")

  def weightsSkat = hc.importTable("src/test/resources/skat.weights",
    types = Map("locus" -> TLocus, "weight" -> TFloat64)).keyBy("locus")

  def vdsSkat: VariantDataset = hc.importVCF("src/test/resources/sample2.vcf")
    .annotateVariantsTable(intervalsSkat, root = "va.genes", product = true)
    .annotateVariantsTable(weightsSkat, root = "va.weight")
    .annotateSamplesTable(phenotypesSkat, root = "sa.pheno0")
    .annotateSamplesTable(covariatesSkat, root = "sa.cov")
    .annotateSamplesExpr("sa.pheno = if (sa.pheno0 == 1.0) false else if (sa.pheno0 == 2.0) true else NA: Boolean")


  @Test def hardcallsBigTest() {

    val useDosages = false
    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val kt = vdsSkat.splitMulti().skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno0", covariates, useDosages)
    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno0", covariates, useDosages)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {

      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }

  @Test def dosagesBigTest() {
    val useDosages = true
    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val kt = vdsSkat.splitMulti().skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno0", covariates, useDosages)
    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno0", covariates, useDosages)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }

  @Test def largeNBigTest() {
    val useDosages = true
    val useLargeN = true
    val useLogistic = false

    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)

    val kt = Skat(vdsSkat.splitMulti(), "gene", "va.genes", false, Option("va.weight"), "sa.pheno0",
              Array("sa.cov.Cov1", "sa.cov.Cov2"),useDosages,useLargeN, useLogistic)

    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno0", covariates, useDosages, useLogistic)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }

  def vdsLogistic: VariantDataset = hc.importVCF("src/test/resources/sample2.vcf")
    .annotateVariantsTable(intervalsSkat, root = "va.genes", product = true)
    .annotateSamplesTable(phenotypesD, root = "sa.pheno")
    .annotateSamplesTable(covariatesSkat, root = "sa.cov")
 //   .annotateSamplesExpr("sa.pheno = if (sa.pheno == 1.0) false else if (sa.pheno == 2.0) true else NA: Boolean")
   // .filterSamplesExpr("sa.pheno.isDefined() && sa.cov.Cov1.isDefined() && sa.cov.Cov2.isDefined()")
    .annotateVariantsExpr("va.AF = gs.callStats(g=> v).AF")
    .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in dbeta(af,1.0,25.0)**2")
    .splitMulti()

  def dosagesBigTestLogistic() {

    val useDosages = true
    val useLargeN = false
    val useLogistic = true

    val resultsArray = testInR(vdsLogistic, "gene", "va.genes", singleKey = false,
      Some("va.weight"), "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages, useLogistic)

    val kt = Skat(vdsLogistic, "gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages, useLargeN, useLogistic)

    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      println(qstat, qstatR)
      println(pval, pvalR)
      println(rows(i).get(3))

      /**
      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
        **/
      i += 1
    }

  }

}


