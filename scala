$su
#cd DSBDAL
#nano input.txt
#spark-shell
>val inputfile = sc.textFile("input.txt")
>val counts = inputfile.flatMap(line=>line.split(" ")).map(word=>(word,1)).reduceByKey(_+_);
>counts.toDebugString
>counts.cache()
>counts.saveAsTextFile("output")

$su
#cd DSBDAL
#cd output.txt
#ls
#cat part-00000
#cat part-00001


#nano Checknum.scala
object Checknum
{
  def main(args: Array[String]): Unit =
  {
    println("enter a number")
    val input = scala.io.StdIn.readLine()
    val num = input.toDouble

    if(num>0)
    {
      println("+ve")
    }
    else if(num<0)
    {
      println("-ve")
    }
    else
    {
      println("its zero")
    }
  }
}

#nano Largestnum.scala
object Largestnum
{
  def main(args: Array[String])=unit
  {
    println("Enter num 1")
    val num1 = scala.io.StdIn.readDouble()

    println("Enter num 2")
    val num2 = scala.io.StdIn.readDouble()

    val largest = if(num1>num2) num1 else num2

    println(s"largest num is : $largest")
    
  }
}

