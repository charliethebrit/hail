package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.kohsuke.args4j.{Option => Args4jOption}

object ShowGlobalAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "showglobals"

  def description = "Shows the signatures for all annotations currently stored in the dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (vds == null)
      fatal("showannotations requires a non-null variant dataset, import or read one first")

    val json = vds.globalSignature.makeJSON(vds.globalAnnotation)
    val result = "Global annotations: `global' = " + pretty(render(json))

    options.output match {
      case null => println(result)
      case path => writeTextFile(path, state.hadoopConf){out => out.write(result)}
    }

    state
  }
}
