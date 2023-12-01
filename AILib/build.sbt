val scala3Version = "3.0.0"

lazy val root = project
  .in(file("."))
  .settings(
    name := "AILib",
    version := "0.1.0",

    scalaVersion := scala3Version,

    libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test"
  )

lazy val macros = project
  .in(file("./macros"))
  .settings(
    name := "macros",
    version := "0.1.0",

    scalaVersion := scala3Version,

    libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test"
  )

lazy val ai = project
  .in(file("./ai"))
  .settings(
    name := "ai",
    version := "0.1.0",

    scalaVersion := scala3Version,

    libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test"
  ).dependsOn(macros)