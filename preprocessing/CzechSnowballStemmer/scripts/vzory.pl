#!/usr/bin/perl -w
use Cwd; #kvuli pouziti getcwd

sub filesInDir($) {
  my $directory = $_[0]; #adresar obsahujici soubory, ktere se budou otvirat
  my $file = "";
  my @files; #do tohoto pole se ukladaji cesty a nazvy souboru v danem adresari

  opendir(DIR, $directory) or die "Nepodarilo se otevrit $directory.";
  while($file = readdir DIR) { #otevre adresar
    #pokud se jedna o soubor && je to textovy soubor
    if((-f $directory."/".$file) && ((-T $directory."/".$file))) {
      #ulozi do @files cestu ke skriptu / zadany adresar / nazev souboru.pripona
      #ulozi pouze soubory s priponou c || cpp || h
      if($file =~ /\.htm$/) {
        push @files, getcwd."/".$directory."/".$file;
      }
    }
  

  }
  closedir DIR;

  return @files;
} #end of filesInDir()

my $progname = $0;

if (scalar @ARGV != 1) {
  print "Usage: $progname <directory>\n";
  exit 1;
}
$dir = shift(@ARGV);

@results = ();
$line = "";
$text = "";
$prev = "";
$konc1pad = "false";
$koncInfinitiv = "false";
@infinitivy = ();
%pok = ();
%bla = ();
%final = ();

my @files = filesInDir($dir); #soubory v adresari $dir
#print @files;
foreach $file (@files) { #otevre kazdy soubor v danem adresari
  open(DESCFILE, $file) or die "Nelze otevrit soubor.";
  $line = "";
  $text = "";

  while($line = <DESCFILE>) {
    $text .= $line;
  }

  close DESCFILE;

  $vzor = "";
  $text =~ /Vzor (.*?)<\/h1>/g;
  $vzor = $1;
  $vzor =~ /-(.+)/;
  $koncInfinitiv = $1;
  push @infinitivy, $koncInfinitiv;
  $vzor =~ s/-//; #odstraneni pomlcky ze vzoru


  @find = $text =~ /}}}(.*?)(\ |<)/g;
  $konc1pad = "false";


  #print $vzor . " - ";
  foreach $res (@find) {
    next if $res eq " ";
    next if $res eq "<";
    next if $res eq "]]";

    if($konc1pad eq "false") {
      $konc1pad = $res;
      if ($konc1pad eq "") {$konc1pad = "X";}
      #print $konc1pad . " ";
    }

    push @results, $res;
    #$pok{$konc1pad} .= " ".$res." ";  #pro podst a prid jmena
    $pok{$koncInfinitiv} .= " ".$res." ";  #pro slovesa

    #print "'" . $res . "' ";
  }
  #print "\n";
}

#pok: klic - prvni pad, value - vsechny koncovky, ktere maji stejny prvni pad
#bla: klic - koncovka,  value - vsechny jeji prvni pady

=komentar
#vypise: 1.pad - vsechny koncovky tohoto padu
while (($k, $v) = each(%pok)) {
  print "$k - $v\n";
}
=cut

#rvu do druheho
while (($k, $v) = each(%pok)) {
  @find = $v =~ /\s(.*?)\s/g;
  foreach $res (@find) {
    $bla{$res} .= " ".$k." ";
  }
}

=komentar
#vypise: koncovka - vsechny prvni pady vzoru, ve kterych se koncovka vyskytuje
while (($k, $v) = each(%bla)) {
  print "$k - $v\n";
}
=cut

#rvu do final
while (($k, $v) = each(%bla)) {
  @find = $v =~ /\s(.*?)\s/g;
  sort(@find);
  $prev = "false";
  foreach $res (@find) {
    next if $res eq $prev;
    $prev = $res;
    $res =~ s/X//;
    $final{$k} .= " '".$res."' ";
  }
}

=jen slovesa
foreach $res (@infinitivy) {
  @tmp = ();
  while (($k, $v) = each(%final)) {
    next if $v ne " '$res' ";
    $pocet = $v =~ s/'/'/g;
    next if $pocet ne 2;  #tiskne jen jednoznacne moznosti koncovka-patricny infinitiv
    next if " '$k' " eq $v;
    push @tmp, "'$k' ";
  }
    print "
    @tmp
    (
      delete
      insert '$res'
    )
    ";
}
=cut

while (($k, $v) = each(%final)) {
  $pocet = $v =~ s/'/'/g;
  #next if $pocet ne 2;  #tiskne jen jednoznacne moznosti koncovka-patricny infinitiv
  next if $pocet eq 2; #tiskne nejednoznacne moznosti
  next if " '$k' " eq $v;
    #print "'$k' "
  print "
  '$k'
  (
    delete
    insert$v
  )
  ";
}


#@results = sort @results;

#foreach $res (@results) {
#  if ($res ne $prev) {
#    print $res . "\n";
#  }
#  $prev = $res;
#}

