package is.hail.methods

import is.hail.utils._
import java.io._

import scala.io.Source
import is.hail.expr._
import is.hail.SparkSuite
import is.hail.io.annotators.IntervalList
import is.hail.variant.{Genotype, VariantDataset}
import is.hail.methods.Skat

import scala.sys.process._
import breeze.linalg._
import is.hail.keytable.KeyTable

import scala.language.postfixOps
import is.hail.methods.Skat.keyedRDDSkat
import is.hail.stats.RegressionUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

case class SkatAggForR[T <: Vector[Double]](xs: ArrayBuilder[T], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {

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
        string = string + separator + A(i, j).toString()
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
    useDosages: Boolean): Array[Row] = {


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

          val rScript = s"Rscript src/test/resources/skatTest.R " +
            s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
            s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
            s"${ uriPath(resultsFile) } " + "C"

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

  def intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")

  def vds: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsTable(intervals, root = "va.genes", product = true)
    .annotateSamplesTable(phenotypes, root = "sa.pheno")
    .annotateSamplesTable(covariates, root = "sa.cov")
    //  .annotateVariantsExpr("va.weight = v.start.toDouble")
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

    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)

    val kt = Skat(vdsSkat.splitMulti(), "gene", "va.genes", false, Option("va.weight"), "sa.pheno0",
              Array("sa.cov.Cov1", "sa.cov.Cov2"),useDosages,useLargeN)

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
}


